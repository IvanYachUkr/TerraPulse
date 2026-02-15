//! TerraPulse feature extraction - Rust PyO3 extension (v3-correct).
//!
//! Goals:
//! - Correctness first (match NumPy-style NaN handling + stable numerics)
//! - Keep Python interface and feature dimensionality stable (224 per cell)
//! - Keep performance reasonable (rayon parallel where it actually helps)
//!
//! Key fixes vs prior versions:
//! - Percentiles: true nanpercentile-style linear interpolation over finite values
//! - Variance/std: stable two-pass accumulation in f64
//! - Tasseled Cap: robust 6-band Sentinel-2 TC (B02,B03,B04,B08,B11,B12)
//! - LBP per-patch: remove artificial zero-corner padding artifacts (edge clamp)

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

const GP: usize = 10;
const N_PX: usize = GP * GP; // 100 pixels per cell
const N_BANDS: usize = 10;
const EPS: f32 = 1e-10;

// LBP parameters
const LBP_P: usize = 8;
const LBP_BINS: usize = LBP_P + 2; // 10 bins: 0..8 uniform, 9 non-uniform

// Band layout (must match Python's order)
const B02: usize = 0;
const B03: usize = 1;
const B04: usize = 2;
const B05: usize = 3;
const B06: usize = 4;
const B07: usize = 5;
const B08: usize = 6;
const B8A: usize = 7;
const B11: usize = 8;
const B12: usize = 9;

// Sentinel-2 Tasseled Cap coefficients (Nedkov, 2017) — 10 bands
// Order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
// Must match Python's TC_BRIGHTNESS / TC_GREENNESS / TC_WETNESS exactly
const TC10_B: [f32; 10] = [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949, 0.1822, 0.0031, 0.1112, 0.0825];
const TC10_G: [f32; 10] = [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.0469, -0.0322, -0.0015, -0.0693, -0.0180];
const TC10_W: [f32; 10] = [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, 0.0882, 0.4572, -0.0021, -0.4064, 0.0117];

// 20m bands that need block-reduce (factor=2) before stats, matching Python V10
const BANDS_20M: [usize; 6] = [B05, B06, B07, B8A, B11, B12];

// Feature counts
const N_BAND_STATS: usize = N_BANDS * 8; // 80
const N_IDX_STATS: usize = 15 * 5;       // 75
const N_TC: usize = 6;                   // 3 components * (mean,std)
const N_SPATIAL: usize = 8;
const N_LBP: usize = 5 * (LBP_BINS + 1); // 55 (10 bins + entropy) * 5
const N_FEAT: usize = N_BAND_STATS + N_IDX_STATS + N_TC + N_SPATIAL + N_LBP; // 224

// =====================================================================
// Utility: reflect indexing for ndimage-like boundary handling
// =====================================================================

#[inline(always)]
fn reflect_index(mut i: isize, len: isize) -> isize {
    if len <= 1 {
        return 0;
    }
    while i < 0 || i >= len {
        if i < 0 {
            i = -i - 1;
        }
        if i >= len {
            i = 2 * len - i - 1;
        }
    }
    i
}

// =====================================================================
// LBP: uniform LUT + bilinear sampling
// =====================================================================

fn build_lbp_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for val in 0u16..256 {
        let v = val as u8;
        let mut transitions = 0u32;
        for i in 0..8u32 {
            let b0 = (v >> i) & 1;
            let b1 = (v >> ((i + 1) % 8)) & 1;
            if b0 != b1 {
                transitions += 1;
            }
        }
        lut[val as usize] = if transitions <= 2 {
            v.count_ones() as u8 // 0..8
        } else {
            (LBP_P + 1) as u8 // non-uniform bin = 9
        };
    }
    lut
}

/// Bilinear interpolation with constant-zero boundary (cval=0).
/// Matches skimage's bilinear_interpolation(&image[0,0], rows, cols, r, c, 'C', 0, &out).
#[inline(always)]
fn bilinear_constant_zero(img: &[f32], h: usize, w: usize, ry: f64, rx: f64) -> f64 {
    let minr = ry.floor() as isize;
    let minc = rx.floor() as isize;
    let maxr = ry.ceil() as isize;
    let maxc = rx.ceil() as isize;
    let dr = ry - minr as f64;
    let dc = rx - minc as f64;

    // get_pixel2d with mode='C', cval=0: out-of-bounds → 0.0
    let get = |r: isize, c: isize| -> f64 {
        if r < 0 || r >= h as isize || c < 0 || c >= w as isize {
            0.0
        } else {
            img[r as usize * w + c as usize] as f64
        }
    };

    let top_left = get(minr, minc);
    let top_right = get(minr, maxc);
    let bottom_left = get(maxr, minc);
    let bottom_right = get(maxr, maxc);

    let top = (1.0 - dc) * top_left + dc * top_right;
    let bottom = (1.0 - dc) * bottom_left + dc * bottom_right;
    (1.0 - dr) * top + dr * bottom
}

fn compute_lbp_raster(img: &[f32], h: usize, w: usize, lut: &[u8; 256]) -> Vec<u8> {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let dr: [f64; 8] = [0.0, -s2, -1.0, -s2, 0.0, s2, 1.0, s2];
    let dc: [f64; 8] = [1.0, s2, 0.0, -s2, -1.0, -s2, 0.0, s2];

    let mut out = vec![0u8; h * w];
    out.par_chunks_mut(w)
        .enumerate()
        .for_each(|(r, row)| {
            let rf = r as f64;
            for c in 0..w {
                let cf = c as f64;
                let center = img[r * w + c] as f64;
                let mut code: u8 = 0;
                for k in 0..8 {
                    let val = bilinear_constant_zero(img, h, w, rf + dr[k], cf + dc[k]);
                    if val >= center {
                        code |= 1 << k;
                    }
                }
                row[c] = lut[code as usize];
            }
        });
    out
}

/// Bilinear interpolation on a GP×GP patch with constant-zero boundary.
/// Matches skimage's bilinear_interpolation(mode='C', cval=0).
#[inline(always)]
fn bilinear_patch_constant_zero(patch: &[f32; N_PX], ry: f64, rx: f64) -> f64 {
    let minr = ry.floor() as isize;
    let minc = rx.floor() as isize;
    let maxr = ry.ceil() as isize;
    let maxc = rx.ceil() as isize;
    let dr = ry - minr as f64;
    let dc = rx - minc as f64;

    let gp = GP as isize;
    let get = |r: isize, c: isize| -> f64 {
        if r < 0 || r >= gp || c < 0 || c >= gp {
            0.0
        } else {
            patch[r as usize * GP + c as usize] as f64
        }
    };

    let top_left = get(minr, minc);
    let top_right = get(minr, maxc);
    let bottom_left = get(maxr, minc);
    let bottom_right = get(maxr, maxc);

    let top = (1.0 - dc) * top_left + dc * top_right;
    let bottom = (1.0 - dc) * bottom_left + dc * bottom_right;
    (1.0 - dr) * top + dr * bottom
}

/// Compute LBP on isolated 10×10 patches with per-cell NaN fill + clip.
/// Matches Python V10's lbp_features(patch_ref) which does:
///   nir = np.where(np.isfinite(nir), nir, np.nanmean(nir))
///   nir = np.clip(nir, 0.0, 1.0)
///   lbp = local_binary_pattern(nir, P=8, R=1, method="uniform")
///
/// `raw_img` is the RAW band data (may contain NaN/non-finite).
/// `clip_01` indicates whether to clip values to [0, 1] (true for spectral bands,
///   false for index images already in [0, 1]).
fn compute_lbp_perpatch(
    raw_img: &[f32],
    h: usize,
    w: usize,
    n_rows: usize,
    n_cols: usize,
    lut: &[u8; 256],
    clip_01: bool,
) -> Vec<u8> {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let dr: [f64; 8] = [0.0, -s2, -1.0, -s2, 0.0, s2, 1.0, s2];
    let dc: [f64; 8] = [1.0, s2, 0.0, -s2, -1.0, -s2, 0.0, s2];

    let n_cells = n_rows * n_cols;

    let cell_codes: Vec<[u8; N_PX]> = (0..n_cells)
        .into_par_iter()
        .map(|ci| {
            let cr = ci / n_cols;
            let cc = ci % n_cols;
            let r0 = cr * GP;
            let c0 = cc * GP;

            // Extract raw patch
            let mut patch = [0.0f32; N_PX];
            for d in 0..GP {
                let src = (r0 + d) * w + c0;
                patch[d * GP..d * GP + GP].copy_from_slice(&raw_img[src..src + GP]);
            }

            // Per-cell NaN fill: nanmean of THIS patch (matches Python exactly)
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for &v in &patch {
                if v.is_finite() {
                    let cv = if clip_01 { v.clamp(0.0, 1.0) } else { v };
                    sum += cv as f64;
                    n += 1;
                }
            }
            let fill = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };

            // Apply NaN fill + clip
            for v in patch.iter_mut() {
                if v.is_finite() {
                    if clip_01 { *v = v.clamp(0.0, 1.0); }
                } else {
                    *v = fill;
                }
            }

            let mut codes = [0u8; N_PX];
            for r in 0..GP {
                for c in 0..GP {
                    let center = patch[r * GP + c] as f64;
                    let mut code: u8 = 0;
                    for k in 0..8 {
                        let val = bilinear_patch_constant_zero(&patch, r as f64 + dr[k], c as f64 + dc[k]);
                        if val >= center {
                            code |= 1 << k;
                        }
                    }
                    codes[r * GP + c] = lut[code as usize];
                }
            }
            codes
        })
        .collect();

    let mut out = vec![0u8; h * w];
    for ci in 0..n_cells {
        let cr = ci / n_cols;
        let cc = ci % n_cols;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for d in 0..GP {
            let dst = (r0 + d) * w + c0;
            out[dst..dst + GP].copy_from_slice(&cell_codes[ci][d * GP..d * GP + GP]);
        }
    }
    out
}

// =====================================================================
// Full-raster convolutions (reflect boundary like ndimage default)
// =====================================================================

fn compute_sobel_mag(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w)
        .enumerate()
        .for_each(|(r, row)| {
            let hh = h as isize;
            let ww = w as isize;
            let rr = r as isize;

            for c in 0..w {
                let cc = c as isize;

                let g = |dr: isize, dc: isize| -> f64 {
                    let r2 = reflect_index(rr + dr, hh) as usize;
                    let c2 = reflect_index(cc + dc, ww) as usize;
                    img[r2 * w + c2] as f64
                };

                // Classic 3x3 Sobel kernels
                let gx = -g(-1, -1) + g(-1, 1)
                    - 2.0 * g(0, -1) + 2.0 * g(0, 1)
                    - g(1, -1) + g(1, 1);

                let gy = -g(-1, -1) - 2.0 * g(-1, 0) - g(-1, 1)
                    + g(1, -1) + 2.0 * g(1, 0) + g(1, 1);

                row[c] = ((gx * gx + gy * gy).sqrt()) as f32;
            }
        });
    out
}

fn compute_laplacian(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w)
        .enumerate()
        .for_each(|(r, row)| {
            let hh = h as isize;
            let ww = w as isize;
            let rr = r as isize;

            for c in 0..w {
                let cc = c as isize;

                let g = |dr: isize, dc: isize| -> f64 {
                    let r2 = reflect_index(rr + dr, hh) as usize;
                    let c2 = reflect_index(cc + dc, ww) as usize;
                    img[r2 * w + c2] as f64
                };

                // 4-neighbor Laplacian: [0 1 0; 1 -4 1; 0 1 0]
                let v = g(-1, 0) + g(1, 0) + g(0, -1) + g(0, 1) - 4.0 * g(0, 0);
                row[c] = v as f32;
            }
        });
    out
}

// =====================================================================
// Image preparation helpers
// =====================================================================

fn clean_band_nan_fill(raw: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for &v in &raw[..h * w] {
        if v.is_finite() {
            sum += v as f64;
            n += 1;
        }
    }
    let fill = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
    raw[..h * w]
        .iter()
        .map(|&v| if v.is_finite() { v } else { fill })
        .collect()
}

/// Same as clean_band_nan_fill but also clips to [0, 1].
/// Matches Python's `_fill_nan(np.clip(band, 0.0, 1.0))` used before LBP.
fn clean_band_nan_fill_clipped(raw: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for &v in &raw[..h * w] {
        if v.is_finite() {
            sum += v.clamp(0.0, 1.0) as f64;
            n += 1;
        }
    }
    let fill = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
    raw[..h * w]
        .iter()
        .map(|&v| if v.is_finite() { v.clamp(0.0, 1.0) } else { fill })
        .collect()
}

#[inline(always)]
fn safe_ratio(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        (a - b) / (a + b + EPS)
    } else {
        f32::NAN
    }
}

// =====================================================================
// Per-cell statistics (nan-aware, stable)
// =====================================================================

#[inline(always)]
fn percentile_linear(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (n as f32 - 1.0) * q;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = pos - lo as f32;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// 8 stats: mean, std, min, max, q25, median, q75, finite_frac
fn cell_stats_8(px: &[f32; N_PX]) -> [f32; 8] {
    let mut vals = [0.0f32; N_PX];
    let mut n: usize = 0;

    let mut sum = 0.0f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;

    for &v in px.iter() {
        if v.is_finite() {
            vals[n] = v;
            n += 1;
            sum += v as f64;
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
    }

    let finite_frac = n as f32 / N_PX as f32;
    if n == 0 {
        return [f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, 0.0];
    }

    let mean = (sum / n as f64) as f32;

    // Stable variance (two-pass) in f64
    let mut var = 0.0f64;
    for i in 0..n {
        let d = vals[i] as f64 - mean as f64;
        var += d * d;
    }
    let std = ((var / n as f64).max(0.0)).sqrt() as f32;

    let vs = &mut vals[..n];
    vs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let q25 = percentile_linear(vs, 0.25);
    let med = percentile_linear(vs, 0.50);
    let q75 = percentile_linear(vs, 0.75);

    [mean, std, mn, mx, q25, med, q75, finite_frac]
}

/// Block-reduce 10×10 → 5×5 via nanmean of 2×2 blocks.
/// Matches Python's _block_reduce_mean(band, factor=2).
fn block_reduce_2x2(px: &[f32; N_PX]) -> ([f32; 25], usize) {
    let mut out = [0.0f32; 25];
    let mut count = 0usize;
    for br in 0..5 {
        for bc in 0..5 {
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for dr in 0..2 {
                for dc in 0..2 {
                    let v = px[(2 * br + dr) * GP + (2 * bc + dc)];
                    if v.is_finite() {
                        sum += v as f64;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out[count] = (sum / n as f64) as f32;
            } else {
                out[count] = f32::NAN;
            }
            count += 1;
        }
    }
    (out, 25)
}

/// 8 stats on a dynamically-sized slice (for block-reduced 25-element arrays)
fn cell_stats_8_dyn(px: &[f32], total_size: usize) -> [f32; 8] {
    let mut vals = Vec::with_capacity(total_size);
    let mut sum = 0.0f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;

    for &v in px.iter().take(total_size) {
        if v.is_finite() {
            vals.push(v);
            sum += v as f64;
            if v < mn { mn = v; }
            if v > mx { mx = v; }
        }
    }

    let n = vals.len();
    let finite_frac = n as f32 / total_size as f32;
    if n == 0 {
        return [f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, 0.0];
    }

    let mean = (sum / n as f64) as f32;
    let mut var = 0.0f64;
    for &v in &vals {
        let d = v as f64 - mean as f64;
        var += d * d;
    }
    let std = ((var / n as f64).max(0.0)).sqrt() as f32;

    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q25 = percentile_linear(&vals, 0.25);
    let med = percentile_linear(&vals, 0.50);
    let q75 = percentile_linear(&vals, 0.75);

    [mean, std, mn, mx, q25, med, q75, finite_frac]
}

#[inline(always)]
fn cell_stats_5(px: &[f32; N_PX]) -> [f32; 5] {
    let s = cell_stats_8(px);
    [s[0], s[1], s[4], s[5], s[6]]
}

#[inline(always)]
fn extract_cell(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; N_PX] {
    let mut px = [0.0f32; N_PX];
    let r0 = cr * GP;
    let c0 = cc * GP;
    for dr in 0..GP {
        let row_off = (r0 + dr) * w + c0;
        px[dr * GP..dr * GP + GP].copy_from_slice(&img[row_off..row_off + GP]);
    }
    px
}

fn cell_lbp_hist(lbp: &[u8], w: usize, cr: usize, cc: usize) -> [f32; LBP_BINS + 1] {
    let mut counts = [0u32; LBP_BINS];
    let r0 = cr * GP;
    let c0 = cc * GP;

    for dr in 0..GP {
        let row_off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let bin = lbp[row_off + dc] as usize;
            if bin < LBP_BINS {
                counts[bin] += 1;
            }
        }
    }

    let inv = 1.0 / N_PX as f32;
    let mut out = [0.0f32; LBP_BINS + 1];
    let mut entropy = 0.0f32;

    for i in 0..LBP_BINS {
        let p = counts[i] as f32 * inv;
        out[i] = p;
        if p > EPS {
            entropy -= p * p.ln();
        }
    }
    out[LBP_BINS] = entropy;
    out
}

fn cell_morans_i(px: &[f32; N_PX]) -> f32 {
    // NaN-aware Moran's I with 4-neighbor pairs (right + down)
    let mut sum = 0.0f64;
    let mut n_valid = 0usize;

    for &v in px.iter() {
        if v.is_finite() {
            sum += v as f64;
            n_valid += 1;
        }
    }
    if n_valid <= 1 {
        return f32::NAN;
    }

    let mean = (sum / n_valid as f64) as f32;

    let mut z = [f32::NAN; N_PX];
    let mut denom = 0.0f64;
    for i in 0..N_PX {
        if px[i].is_finite() {
            let dv = px[i] - mean;
            z[i] = dv;
            denom += (dv as f64) * (dv as f64);
        }
    }
    if denom < 1e-12 {
        return 0.0;
    }

    let mut w_sum = 0.0f64;
    let mut n_pairs = 0usize;

    for r in 0..GP {
        for c in 0..GP {
            let i = r * GP + c;
            if !z[i].is_finite() {
                continue;
            }
            if c + 1 < GP && z[i + 1].is_finite() {
                w_sum += (z[i] as f64) * (z[i + 1] as f64);
                n_pairs += 1;
            }
            if r + 1 < GP && z[i + GP].is_finite() {
                w_sum += (z[i] as f64) * (z[i + GP] as f64);
                n_pairs += 1;
            }
        }
    }

    if n_pairs == 0 {
        return 0.0;
    }

    ((n_valid as f64 / n_pairs as f64) * w_sum / denom) as f32
}

fn cell_agg_3(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; 3] {
    let r0 = cr * GP;
    let c0 = cc * GP;

    let mut sum = 0.0f64;
    let mut mx = f32::NEG_INFINITY;

    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let v = img[off + dc];
            sum += v as f64;
            if v > mx {
                mx = v;
            }
        }
    }

    let mean = sum / N_PX as f64;

    let mut var = 0.0f64;
    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let d = img[off + dc] as f64 - mean;
            var += d * d;
        }
    }
    let std = ((var / N_PX as f64).max(0.0)).sqrt() as f32;

    [mean as f32, std, mx]
}

fn cell_lap_stats(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; 2] {
    let r0 = cr * GP;
    let c0 = cc * GP;

    let mut abs_sum = 0.0f64;
    let mut sum = 0.0f64;

    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let v = img[off + dc] as f64;
            abs_sum += v.abs();
            sum += v;
        }
    }

    let mean = sum / N_PX as f64;

    let mut var = 0.0f64;
    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let d = img[off + dc] as f64 - mean;
            var += d * d;
        }
    }

    [(abs_sum / N_PX as f64) as f32, ((var / N_PX as f64).max(0.0)).sqrt() as f32]
}

// =====================================================================
// Main extraction
// =====================================================================

fn extract_cell_features(
    spec: &[f32],
    h: usize,
    w: usize,
    cr: usize,
    cc: usize,
    sobel: &[f32],
    lap: &[f32],
    nir_clean: &[f32],
    lbp_nir: &[u8],
    lbp_ndvi: &[u8],
    lbp_evi2: &[u8],
    lbp_swir1: &[u8],
    lbp_ndti: &[u8],
) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let mut fi: usize = 0;

    // 1) Band stats (80)
    // 20m bands (B05/B06/B07/B8A/B11/B12) get block-reduced 10×10→5×5
    // before stats, matching Python V10's _block_reduce_mean(band, factor=2)
    let mut band_px = [[0.0f32; N_PX]; N_BANDS];
    for b in 0..N_BANDS {
        let band_off = b * h * w;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for dr in 0..GP {
            let src_off = band_off + (r0 + dr) * w + c0;
            let dst_off = dr * GP;
            band_px[b][dst_off..dst_off + GP].copy_from_slice(&spec[src_off..src_off + GP]);
        }
        let is_20m = BANDS_20M.contains(&b);
        let s = if is_20m {
            let (reduced, n) = block_reduce_2x2(&band_px[b]);
            cell_stats_8_dyn(&reduced, n)
        } else {
            cell_stats_8(&band_px[b])
        };
        for v in s {
            out[fi] = v;
            fi += 1;
        }
    }

    // 2) Indices (75)
    let blue = &band_px[B02];
    let green = &band_px[B03];
    let red = &band_px[B04];
    let re1 = &band_px[B05];
    let re2 = &band_px[B06];
    let re3 = &band_px[B07];
    let nir = &band_px[B08];
    let swir1 = &band_px[B11];
    let swir2 = &band_px[B12];

    let mut idx_px = [0.0f32; N_PX];

    // 10 normalized differences
    let pairs: [(usize, usize); 10] = [
        (B08, B04), // NDVI
        (B03, B08), // NDWI
        (B11, B08), // NDBI
        (B08, B11), // NDMI
        (B08, B12), // NBR
        (B08, B05), // NDRE1
        (B08, B06), // NDRE2
        (B03, B11), // MNDWI
        (B08, B03), // GNDVI
        (B11, B12), // NDTI
    ];

    let mut ndvi_px = [0.0f32; N_PX];
    for (pi, &(a, b)) in pairs.iter().enumerate() {
        for i in 0..N_PX {
            idx_px[i] = safe_ratio(band_px[a][i], band_px[b][i]);
        }
        if pi == 0 {
            ndvi_px = idx_px;
        }
        let s = cell_stats_5(&idx_px);
        for v in s {
            out[fi] = v;
            fi += 1;
        }
    }

    // SAVI
    for i in 0..N_PX {
        idx_px[i] = if nir[i].is_finite() && red[i].is_finite() {
            1.5 * (nir[i] - red[i]) / (nir[i] + red[i] + 0.5 + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // BSI
    for i in 0..N_PX {
        idx_px[i] = if swir1[i].is_finite() && red[i].is_finite() && nir[i].is_finite() && blue[i].is_finite() {
            let num = (swir1[i] + red[i]) - (nir[i] + blue[i]);
            num / ((swir1[i] + red[i]) + (nir[i] + blue[i]) + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // EVI2
    for i in 0..N_PX {
        idx_px[i] = if nir[i].is_finite() && red[i].is_finite() {
            2.5 * (nir[i] - red[i]) / (nir[i] + 2.4 * red[i] + 1.0 + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // IRECI
    for i in 0..N_PX {
        idx_px[i] = if re3[i].is_finite() && red[i].is_finite() && re1[i].is_finite() && re2[i].is_finite() {
            (re3[i] - red[i]) / (re1[i] / (re2[i] + EPS) + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // CRI1
    for i in 0..N_PX {
        idx_px[i] = if green[i].is_finite() && re1[i].is_finite() && green[i] > EPS && re1[i] > EPS {
            (1.0 / green[i]) - (1.0 / re1[i])
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // 3) Tasseled Cap (6) — 10-band Nedkov 2017, matching Python exactly
    // dot product of all 10 bands with each TC coefficient vector
    for coeff in [TC10_B, TC10_G, TC10_W] {
        let mut vals = [0.0f32; N_PX];
        let mut n = 0usize;

        for i in 0..N_PX {
            let mut ok = true;
            let mut dot = 0.0f32;
            for b in 0..N_BANDS {
                let v = band_px[b][i];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                dot += v * coeff[b];
            }
            if ok {
                vals[n] = dot;
                n += 1;
            }
        }

        if n == 0 {
            out[fi] = f32::NAN;
            out[fi + 1] = f32::NAN;
        } else {
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += vals[i] as f64;
            }
            let mean = (sum / n as f64) as f32;

            let mut var = 0.0f64;
            for i in 0..n {
                let d = vals[i] as f64 - mean as f64;
                var += d * d;
            }
            let std = ((var / n as f64).max(0.0)).sqrt() as f32;

            out[fi] = mean;
            out[fi + 1] = std;
        }
        fi += 2;
    }

    // 4) Spatial (8)
    let e = cell_agg_3(sobel, w, cr, cc);
    out[fi] = e[0]; fi += 1;
    out[fi] = e[1]; fi += 1;
    out[fi] = e[2]; fi += 1;

    let l = cell_lap_stats(lap, w, cr, cc);
    out[fi] = l[0]; fi += 1;
    out[fi] = l[1]; fi += 1;

    let nir_px = extract_cell(nir_clean, w, cr, cc);
    out[fi] = cell_morans_i(&nir_px); fi += 1;

    let ndvi_s = cell_stats_8(&ndvi_px);
    out[fi] = ndvi_s[3] - ndvi_s[2]; fi += 1; // range
    out[fi] = ndvi_s[6] - ndvi_s[4]; fi += 1; // IQR

    // 5) Multi-band LBP (55)
    let lbp_imgs = [lbp_nir, lbp_ndvi, lbp_evi2, lbp_swir1, lbp_ndti];
    for lbp in lbp_imgs {
        let hst = cell_lbp_hist(lbp, w, cr, cc);
        for v in hst {
            out[fi] = v;
            fi += 1;
        }
    }

    debug_assert_eq!(fi, N_FEAT);
    out
}

// =====================================================================
// Python interface
// =====================================================================

#[pyfunction]
fn extract_season<'py>(
    py: Python<'py>,
    spectral: PyReadonlyArray3<'py, f32>,
    valid_frac: PyReadonlyArray2<'py, f32>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let spec_view = spectral.as_array();
    let _vf = valid_frac.as_array(); // kept for interface compatibility

    let h = spec_view.shape()[1];
    let w = spec_view.shape()[2];
    assert_eq!(spec_view.shape()[0], N_BANDS);
    assert_eq!(h, n_rows * GP);
    assert_eq!(w, n_cols * GP);

    let spec: Vec<f32>;
    let spec_slice: &[f32] = match spec_view.as_slice() {
        Some(s) => s,
        None => {
            spec = spec_view.iter().copied().collect();
            &spec
        }
    };

    let band_slice = |b: usize| -> &[f32] { &spec_slice[b * h * w..(b + 1) * h * w] };

    let lbp_lut = build_lbp_lut();

    // Phase 1: spatial features use unclipped NIR
    let nir_clean = clean_band_nan_fill(band_slice(B08), h, w);
    let sobel = compute_sobel_mag(&nir_clean, h, w);
    let laplacian = compute_laplacian(&nir_clean, h, w);

    let red_clean = clean_band_nan_fill(band_slice(B04), h, w);
    let swir1_clean = clean_band_nan_fill(band_slice(B11), h, w);
    let swir2_clean = clean_band_nan_fill(band_slice(B12), h, w);

    // LBP inputs: clipped to [0,1] to match Python's np.clip(band, 0, 1)
    let nir_lbp = clean_band_nan_fill_clipped(band_slice(B08), h, w);
    let swir1_lbp = clean_band_nan_fill_clipped(band_slice(B11), h, w);

    // Index images for LBP (rescaled to [0,1])
    let ndvi_img: Vec<f32> = (0..h * w)
        .into_par_iter()
        .map(|i| {
            let v = (nir_clean[i] - red_clean[i]) / (nir_clean[i] + red_clean[i] + EPS);
            ((v + 1.0) * 0.5).clamp(0.0, 1.0)
        })
        .collect();

    let evi2_img: Vec<f32> = (0..h * w)
        .into_par_iter()
        .map(|i| {
            let e = 2.5 * (nir_clean[i] - red_clean[i]) / (nir_clean[i] + 2.4 * red_clean[i] + 1.0 + EPS);
            ((e + 0.5) / 1.5).clamp(0.0, 1.0)
        })
        .collect();

    let ndti_img: Vec<f32> = (0..h * w)
        .into_par_iter()
        .map(|i| {
            let v = (swir1_clean[i] - swir2_clean[i]) / (swir1_clean[i] + swir2_clean[i] + EPS);
            ((v + 1.0) * 0.5).clamp(0.0, 1.0)
        })
        .collect();

    let lbp_nir = compute_lbp_raster(&nir_lbp, h, w, &lbp_lut);
    let lbp_ndvi = compute_lbp_raster(&ndvi_img, h, w, &lbp_lut);
    let lbp_evi2 = compute_lbp_raster(&evi2_img, h, w, &lbp_lut);
    let lbp_swir1 = compute_lbp_raster(&swir1_lbp, h, w, &lbp_lut);
    let lbp_ndti = compute_lbp_raster(&ndti_img, h, w, &lbp_lut);

    // Phase 2
    let n_cells = n_rows * n_cols;
    let results: Vec<[f32; N_FEAT]> = (0..n_cells)
        .into_par_iter()
        .map(|ci| {
            let cr = ci / n_cols;
            let cc = ci % n_cols;
            extract_cell_features(
                spec_slice, h, w, cr, cc,
                &sobel, &laplacian, &nir_clean,
                &lbp_nir, &lbp_ndvi, &lbp_evi2, &lbp_swir1, &lbp_ndti,
            )
        })
        .collect();

    let mut flat = Vec::with_capacity(n_cells * N_FEAT);
    for feats in &results {
        flat.extend_from_slice(feats);
    }

    Ok(ndarray::Array1::from_vec(flat).into_pyarray(py).into())
}

#[pyfunction]
fn n_features_per_cell() -> usize {
    N_FEAT
}

#[pyfunction]
fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FEAT);

    let bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"];
    let bst = ["mean","std","min","max","q25","median","q75","finite_frac"];
    for bn in &bands {
        for sn in &bst {
            names.push(format!("{bn}_{sn}"));
        }
    }

    let idxs = [
        "NDVI","NDWI","NDBI","NDMI","NBR","NDRE1","NDRE2",
        "MNDWI","GNDVI","NDTI","SAVI","BSI","EVI2","IRECI","CRI1"
    ];
    let ist = ["mean","std","q25","median","q75"];
    for idn in &idxs {
        for sn in &ist {
            names.push(format!("{idn}_{sn}"));
        }
    }

    for tc in &["TC_bright","TC_green","TC_wet"] {
        names.push(format!("{tc}_mean"));
        names.push(format!("{tc}_std"));
    }

    names.extend(
        ["edge_mean","edge_std","edge_max","lap_abs_mean","lap_std","morans_I_NIR","NDVI_range","NDVI_iqr"]
            .iter()
            .map(|s| s.to_string()),
    );

    // NIR LBP: use "LBP_u8_X" to match Python V10 naming exactly
    for b in 0..LBP_BINS {
        names.push(format!("LBP_u{LBP_P}_{b}"));
    }
    names.push("LBP_entropy".to_string());
    // Other LBP bands keep their band-prefixed names
    for lb in &["NDVI","EVI2","SWIR1","NDTI"] {
        for b in 0..LBP_BINS {
            names.push(format!("LBP_{lb}_u{LBP_P}_{b}"));
        }
        names.push(format!("LBP_{lb}_entropy"));
    }

    assert_eq!(names.len(), N_FEAT);
    names
}

#[pyfunction]
fn feature_names_suffixed(suffixes: Vec<String>) -> Vec<String> {
    let base = feature_names();
    let mut out = Vec::with_capacity(base.len() * suffixes.len());
    for suf in &suffixes {
        for name in &base {
            out.push(format!("{name}_{suf}"));
        }
    }
    out
}

#[pyfunction]
fn extract_all_seasons<'py>(
    py: Python<'py>,
    spectral_list: Vec<PyReadonlyArray3<'py, f32>>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n_seasons = spectral_list.len();
    let n_cells = n_rows * n_cols;
    let total_feats = n_cells * n_seasons * N_FEAT;

    let mut season_data: Vec<Vec<f32>> = Vec::with_capacity(n_seasons);
    let mut h = 0usize;
    let mut w = 0usize;

    for (si, spec_arr) in spectral_list.iter().enumerate() {
        let view = spec_arr.as_array();
        if si == 0 {
            h = view.shape()[1];
            w = view.shape()[2];
            assert_eq!(view.shape()[0], N_BANDS);
            assert_eq!(h, n_rows * GP);
            assert_eq!(w, n_cols * GP);
        }
        let data: Vec<f32> = match view.as_slice() {
            Some(s) => s.to_vec(),
            None => view.iter().copied().collect(),
        };
        season_data.push(data);
    }

    let lbp_lut = build_lbp_lut();

    let season_results: Vec<Vec<[f32; N_FEAT]>> = season_data
        .iter()
        .map(|spec_slice| {
            let band_slice = |b: usize| -> &[f32] { &spec_slice[b * h * w..(b + 1) * h * w] };

            let nir_clean = clean_band_nan_fill(band_slice(B08), h, w);
            let sobel = compute_sobel_mag(&nir_clean, h, w);
            let laplacian = compute_laplacian(&nir_clean, h, w);

            let red_clean = clean_band_nan_fill(band_slice(B04), h, w);
            let swir1_clean = clean_band_nan_fill(band_slice(B11), h, w);
            let swir2_clean = clean_band_nan_fill(band_slice(B12), h, w);

            // LBP inputs: clipped to [0,1] to match Python
            let nir_lbp = clean_band_nan_fill_clipped(band_slice(B08), h, w);
            let swir1_lbp = clean_band_nan_fill_clipped(band_slice(B11), h, w);

            let ndvi_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let v = (nir_clean[i] - red_clean[i]) / (nir_clean[i] + red_clean[i] + EPS);
                    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
                })
                .collect();

            let evi2_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let e = 2.5 * (nir_clean[i] - red_clean[i]) / (nir_clean[i] + 2.4 * red_clean[i] + 1.0 + EPS);
                    ((e + 0.5) / 1.5).clamp(0.0, 1.0)
                })
                .collect();

            let ndti_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let v = (swir1_clean[i] - swir2_clean[i]) / (swir1_clean[i] + swir2_clean[i] + EPS);
                    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
                })
                .collect();

            let lbp_nir = compute_lbp_raster(&nir_lbp, h, w, &lbp_lut);
            let lbp_ndvi = compute_lbp_raster(&ndvi_img, h, w, &lbp_lut);
            let lbp_evi2 = compute_lbp_raster(&evi2_img, h, w, &lbp_lut);
            let lbp_swir1 = compute_lbp_raster(&swir1_lbp, h, w, &lbp_lut);
            let lbp_ndti = compute_lbp_raster(&ndti_img, h, w, &lbp_lut);

            (0..n_cells)
                .into_par_iter()
                .map(|ci| {
                    extract_cell_features(
                        spec_slice, h, w, ci / n_cols, ci % n_cols,
                        &sobel, &laplacian, &nir_clean,
                        &lbp_nir, &lbp_ndvi, &lbp_evi2, &lbp_swir1, &lbp_ndti,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut flat = vec![0.0f32; total_feats];
    for ci in 0..n_cells {
        let cell_base = ci * n_seasons * N_FEAT;
        for si in 0..n_seasons {
            let dst = cell_base + si * N_FEAT;
            flat[dst..dst + N_FEAT].copy_from_slice(&season_results[si][ci]);
        }
    }

    Ok(ndarray::Array1::from_vec(flat).into_pyarray(py).into())
}

#[pyfunction]
fn extract_all_seasons_v2<'py>(
    py: Python<'py>,
    spectral_list: Vec<PyReadonlyArray3<'py, f32>>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n_seasons = spectral_list.len();
    let n_cells = n_rows * n_cols;
    let total_feats = n_cells * n_seasons * N_FEAT;

    let mut season_data: Vec<Vec<f32>> = Vec::with_capacity(n_seasons);
    let mut h = 0usize;
    let mut w = 0usize;

    for (si, spec_arr) in spectral_list.iter().enumerate() {
        let view = spec_arr.as_array();
        if si == 0 {
            h = view.shape()[1];
            w = view.shape()[2];
            assert_eq!(view.shape()[0], N_BANDS);
            assert_eq!(h, n_rows * GP);
            assert_eq!(w, n_cols * GP);
        }
        let data: Vec<f32> = match view.as_slice() {
            Some(s) => s.to_vec(),
            None => view.iter().copied().collect(),
        };
        season_data.push(data);
    }

    let lbp_lut = build_lbp_lut();

    let season_results: Vec<Vec<[f32; N_FEAT]>> = season_data
        .iter()
        .map(|spec_slice| {
            let band_slice = |b: usize| -> &[f32] { &spec_slice[b * h * w..(b + 1) * h * w] };

            let nir_clean = clean_band_nan_fill(band_slice(B08), h, w);
            let sobel = compute_sobel_mag(&nir_clean, h, w);
            let laplacian = compute_laplacian(&nir_clean, h, w);

            let red_clean = clean_band_nan_fill(band_slice(B04), h, w);
            let swir1_clean = clean_band_nan_fill(band_slice(B11), h, w);
            let swir2_clean = clean_band_nan_fill(band_slice(B12), h, w);

            // LBP inputs: clipped to [0,1] to match Python
            let nir_lbp = clean_band_nan_fill_clipped(band_slice(B08), h, w);
            let swir1_lbp = clean_band_nan_fill_clipped(band_slice(B11), h, w);

            let ndvi_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let v = (nir_clean[i] - red_clean[i]) / (nir_clean[i] + red_clean[i] + EPS);
                    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
                })
                .collect();

            let evi2_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let e = 2.5 * (nir_clean[i] - red_clean[i]) / (nir_clean[i] + 2.4 * red_clean[i] + 1.0 + EPS);
                    ((e + 0.5) / 1.5).clamp(0.0, 1.0)
                })
                .collect();

            let ndti_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let v = (swir1_clean[i] - swir2_clean[i]) / (swir1_clean[i] + swir2_clean[i] + EPS);
                    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
                })
                .collect();

            // Per-patch LBP: isolated 10x10 patches with per-cell NaN fill
            // NIR/SWIR1: pass raw band data, clip_01=true (per-cell NaN fill + clip)
            let lbp_nir = compute_lbp_perpatch(band_slice(B08), h, w, n_rows, n_cols, &lbp_lut, true);
            let lbp_swir1 = compute_lbp_perpatch(band_slice(B11), h, w, n_rows, n_cols, &lbp_lut, true);
            // Index images: already NaN-free and in [0,1], no clipping needed
            let lbp_ndvi = compute_lbp_perpatch(&ndvi_img, h, w, n_rows, n_cols, &lbp_lut, false);
            let lbp_evi2 = compute_lbp_perpatch(&evi2_img, h, w, n_rows, n_cols, &lbp_lut, false);
            let lbp_ndti = compute_lbp_perpatch(&ndti_img, h, w, n_rows, n_cols, &lbp_lut, false);

            (0..n_cells)
                .into_par_iter()
                .map(|ci| {
                    extract_cell_features(
                        spec_slice, h, w, ci / n_cols, ci % n_cols,
                        &sobel, &laplacian, &nir_clean,
                        &lbp_nir, &lbp_ndvi, &lbp_evi2, &lbp_swir1, &lbp_ndti,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut flat = vec![0.0f32; total_feats];
    for ci in 0..n_cells {
        let cell_base = ci * n_seasons * N_FEAT;
        for si in 0..n_seasons {
            let dst = cell_base + si * N_FEAT;
            flat[dst..dst + N_FEAT].copy_from_slice(&season_results[si][ci]);
        }
    }

    Ok(ndarray::Array1::from_vec(flat).into_pyarray(py).into())
}

#[pymodule]
fn terrapulse_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_season, m)?)?;
    m.add_function(wrap_pyfunction!(extract_all_seasons, m)?)?;
    m.add_function(wrap_pyfunction!(extract_all_seasons_v2, m)?)?;
    m.add_function(wrap_pyfunction!(n_features_per_cell, m)?)?;
    m.add_function(wrap_pyfunction!(feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(feature_names_suffixed, m)?)?;
    Ok(())
}
