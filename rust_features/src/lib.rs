//! TerraPulse feature extraction — Rust PyO3 extension (v2).
//!
//! Architecture: two-phase pipeline for maximum correctness & speed.
//!   Phase 1: Full-raster pre-computation (row-parallel via rayon)
//!            — Sobel, Laplacian, LBP with bilinear interpolation
//!   Phase 2: Per-cell aggregation (cell-parallel via rayon)
//!            — Band stats, index stats, TC, spatial stats, LBP histograms
//!
//! Matches Python's skimage/scipy algorithms:
//!   - LBP: bilinear interpolation for sub-pixel neighbors (P=8, R=1)
//!   - Sobel: 3×3 kernel on full raster (not per-cell)
//!   - Percentiles: NaN→+inf, sort all 100, index at [24, 49, 74]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

const GP: usize = 10;
const N_PX: usize = GP * GP;  // 100 pixels per cell
const EPS: f32 = 1e-10;
const N_BANDS: usize = 10;

// LBP parameters
const LBP_P: usize = 8;
const LBP_BINS: usize = LBP_P + 2; // 10 uniform + 1 non-uniform = but actually P+2 = 10

// Percentile indices (for 100-element sorted arrays)
const Q25: usize = 24;
const Q50: usize = 49;
const Q75: usize = 74;

// Band layout (must match Python's BAND_NAMES order)
const B02: usize = 0;
const B03: usize = 1;
const B04: usize = 2;
const B05: usize = 3;
const B06: usize = 4;
const B07: usize = 5;
const B08: usize = 6;
// B8A = 7
const B11: usize = 8;
const B12: usize = 9;

// Tasseled Cap coefficients
const TC_B: [f32; 10] = [0.3510,0.3813,0.3437,0.7196,0.2396,0.1949,0.1822,0.0031,0.1112,0.0825];
const TC_G: [f32; 10] = [-0.3599,-0.3533,-0.4734,0.6633,0.0087,-0.0469,-0.0322,-0.0015,-0.0693,-0.0180];
const TC_W: [f32; 10] = [0.2578,0.2305,0.0883,0.1071,-0.7611,0.0882,0.4572,-0.0021,-0.4064,0.0117];

// Feature counts
const N_BAND_STATS: usize = N_BANDS * 8;    // 80
const N_IDX_STATS: usize = 15 * 5;          // 75
const N_TC: usize = 6;
const N_SPATIAL: usize = 8;
const N_LBP: usize = 5 * (LBP_BINS + 1);   // 55
const N_FEAT: usize = N_BAND_STATS + N_IDX_STATS + N_TC + N_SPATIAL + N_LBP; // 224

// =====================================================================
// LBP: bilinear interpolation matching skimage
// =====================================================================

/// Build uniform LBP lookup table: bit pattern → bin index.
fn build_lbp_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for val in 0u16..256 {
        let v = val as u8;
        let mut transitions = 0u32;
        for i in 0..8u32 {
            let b0 = (v >> i) & 1;
            let b1 = (v >> ((i + 1) % 8)) & 1;
            if b0 != b1 { transitions += 1; }
        }
        lut[val as usize] = if transitions <= 2 {
            v.count_ones() as u8
        } else {
            (LBP_P + 1) as u8 // non-uniform bin
        };
    }
    lut
}

/// Bilinear interpolation at sub-pixel (ry, rx), clamped boundaries.
/// Uses f64 arithmetic to match skimage's Cython (double precision).
#[inline(always)]
fn bilinear(img: &[f32], h: usize, w: usize, ry: f64, rx: f64) -> f64 {
    let fy = ry.floor() as i64;
    let fx = rx.floor() as i64;
    let ty = ry - fy as f64;
    let tx = rx - fx as f64;
    let r0 = fy.clamp(0, h as i64 - 1) as usize;
    let r1 = (fy + 1).clamp(0, h as i64 - 1) as usize;
    let c0 = fx.clamp(0, w as i64 - 1) as usize;
    let c1 = (fx + 1).clamp(0, w as i64 - 1) as usize;
    let v00 = img[r0 * w + c0] as f64;
    let v01 = img[r0 * w + c1] as f64;
    let v10 = img[r1 * w + c0] as f64;
    let v11 = img[r1 * w + c1] as f64;
    (1.0 - ty) * ((1.0 - tx) * v00 + tx * v01) + ty * ((1.0 - tx) * v10 + tx * v11)
}

/// Compute LBP codes for entire raster (row-parallel).
/// Matches skimage's local_binary_pattern(P=8, R=1, method="uniform").
fn compute_lbp_raster(img: &[f32], h: usize, w: usize, lut: &[u8; 256]) -> Vec<u8> {
    // Pre-compute neighbor offsets (P=8, R=1)
    // r_p = -R * sin(2π·k/P),  c_p = R * cos(2π·k/P)
    let s2 = std::f64::consts::FRAC_1_SQRT_2; // 0.7071...
    let dr: [f64; 8] = [0.0, -s2, -1.0, -s2, 0.0, s2, 1.0, s2];
    let dc: [f64; 8] = [1.0, s2, 0.0, -s2, -1.0, -s2, 0.0, s2];

    let mut out = vec![0u8; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        let rf = r as f64;
        for c in 0..w {
            let cf = c as f64;
            let center = img[r * w + c] as f64;
            let mut code: u8 = 0;
            for k in 0..8 {
                let ry = rf + dr[k];
                let rx = cf + dc[k];
                let val = bilinear(img, h, w, ry, rx);
                if val >= center {
                    code |= 1 << k;
                }
            }
            row[c] = lut[code as usize];
        }
    });
    out
}

// =====================================================================
// Full-raster convolutions (row-parallel)
// =====================================================================

/// Sobel edge magnitude on full raster. Matches scipy.ndimage.sobel.
fn compute_sobel_mag(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        for c in 0..w {
            let g = |dr: i32, dc: i32| -> f32 {
                let rr = (r as i32 + dr).clamp(0, h as i32 - 1) as usize;
                let cc = (c as i32 + dc).clamp(0, w as i32 - 1) as usize;
                img[rr * w + cc]
            };
            let gx = -g(-1,-1) + g(-1,1) - 2.0*g(0,-1) + 2.0*g(0,1) - g(1,-1) + g(1,1);
            let gy = -g(-1,-1) - 2.0*g(-1,0) - g(-1,1) + g(1,-1) + 2.0*g(1,0) + g(1,1);
            row[c] = (gx * gx + gy * gy).sqrt();
        }
    });
    out
}

/// Laplacian on full raster. Kernel: [[0,1,0],[1,-4,1],[0,1,0]].
fn compute_laplacian(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        for c in 0..w {
            let g = |dr: i32, dc: i32| -> f32 {
                let rr = (r as i32 + dr).clamp(0, h as i32 - 1) as usize;
                let cc = (c as i32 + dc).clamp(0, w as i32 - 1) as usize;
                img[rr * w + cc]
            };
            row[c] = g(-1,0) + g(1,0) + g(0,-1) + g(0,1) - 4.0 * g(0,0);
        }
    });
    out
}

// =====================================================================
// Image preparation helpers
// =====================================================================

/// NaN-fill a band image: replace NaN with global mean, clamp to [0, 1].
fn clean_band(raw: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for &v in &raw[..h * w] {
        if v.is_finite() { sum += v as f64; n += 1; }
    }
    let fill = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
    raw[..h * w].iter().map(|&v| {
        if v.is_finite() { v.clamp(0.0, 1.0) } else { fill }
    }).collect()
}

/// Normalised difference: (a - b) / (a + b + eps), NaN if either input is NaN.
#[inline(always)]
fn safe_ratio(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        (a - b) / (a + b + EPS)
    } else { f32::NAN }
}

// =====================================================================
// Per-cell statistics (matching Python's _fast_percentiles)
// =====================================================================

/// 8 stats: mean, std, min, max, q25, median, q75, finite_frac.
/// Percentiles: NaN→+inf, sort all 100, index at fixed positions.
fn cell_stats_8(px: &[f32; N_PX]) -> [f32; 8] {
    let mut n = 0u32;
    let mut sum = 0.0f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;

    for &v in px.iter() {
        if v.is_finite() {
            n += 1;
            sum += v as f64;
            if v < mn { mn = v; }
            if v > mx { mx = v; }
        }
    }
    if n == 0 { return [f32::NAN; 8]; }

    let mean = (sum / n as f64) as f32;
    let mut var = 0.0f64;
    for &v in px.iter() {
        if v.is_finite() {
            let d = (v - mean) as f64;
            var += d * d;
        }
    }
    let std = ((var / n as f64) as f32).sqrt();

    // Sort with NaN→+inf (matches Python _fast_percentiles)
    let mut sorted = *px;
    for v in sorted.iter_mut() {
        if !v.is_finite() { *v = f32::INFINITY; }
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let q25 = if sorted[Q25].is_finite() { sorted[Q25] } else { f32::NAN };
    let med = if sorted[Q50].is_finite() { sorted[Q50] } else { f32::NAN };
    let q75 = if sorted[Q75].is_finite() { sorted[Q75] } else { f32::NAN };

    [mean, std, mn, mx, q25, med, q75, n as f32 / N_PX as f32]
}

/// 5 stats: mean, std, q25, median, q75.
#[inline]
fn cell_stats_5(px: &[f32; N_PX]) -> [f32; 5] {
    let s = cell_stats_8(px);
    [s[0], s[1], s[4], s[5], s[6]]
}

/// Extract cell pixels from a flat (H×W) image.
#[inline]
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

/// LBP histogram + entropy from pre-computed LBP codes.
fn cell_lbp_hist(lbp: &[u8], w: usize, cr: usize, cc: usize) -> [f32; LBP_BINS + 1] {
    let mut counts = [0u32; LBP_BINS];
    let r0 = cr * GP;
    let c0 = cc * GP;
    for dr in 0..GP {
        let row_off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            counts[lbp[row_off + dc] as usize] += 1;
        }
    }
    let mut out = [0.0f32; LBP_BINS + 1];
    let inv = 1.0 / N_PX as f32;
    let mut entropy = 0.0f32;
    for i in 0..LBP_BINS {
        let p = counts[i] as f32 * inv;
        out[i] = p;
        if p > EPS { entropy -= p * p.ln(); }
    }
    out[LBP_BINS] = entropy;
    out
}

/// Moran's I for a cell patch.
fn cell_morans_i(px: &[f32; N_PX]) -> f32 {
    let sum: f32 = px.iter().sum();
    let mean = sum / N_PX as f32;
    let mut denom = 0.0f32;
    let mut w_sum = 0.0f32;
    let mut z = [0.0f32; N_PX];
    for i in 0..N_PX {
        z[i] = px[i] - mean;
        denom += z[i] * z[i];
    }
    if denom < 1e-10 { return 0.0; }
    for r in 0..GP {
        for c in 0..GP {
            let i = r * GP + c;
            if c + 1 < GP { w_sum += z[i] * z[i + 1]; }
            if r + 1 < GP { w_sum += z[i] * z[i + GP]; }
        }
    }
    let n_pairs = (GP * (GP - 1) * 2) as f32;
    (N_PX as f32 / n_pairs) * w_sum / denom
}

/// Mean, std, max over cell pixels from pre-computed raster.
fn cell_agg_3(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; 3] {
    let r0 = cr * GP;
    let c0 = cc * GP;
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut mx = f32::NEG_INFINITY;
    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let v = img[off + dc];
            sum += v;
            sum2 += v * v;
            if v > mx { mx = v; }
        }
    }
    let mean = sum / N_PX as f32;
    let std = ((sum2 / N_PX as f32) - mean * mean).max(0.0).sqrt();
    [mean, std, mx]
}

/// Abs-mean and std for Laplacian cell.
fn cell_lap_stats(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; 2] {
    let r0 = cr * GP;
    let c0 = cc * GP;
    let mut abs_sum = 0.0f32;
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let v = img[off + dc];
            abs_sum += v.abs();
            sum += v;
            sum2 += v * v;
        }
    }
    let mean = sum / N_PX as f32;
    let std = ((sum2 / N_PX as f32) - mean * mean).max(0.0).sqrt();
    [abs_sum / N_PX as f32, std]
}

// =====================================================================
// Main extraction
// =====================================================================

/// Extract all 224 features for one cell.
fn extract_cell_features(
    // Raw spectral (pre-normalised), flat (N_BANDS * H * W)
    spec: &[f32], h: usize, w: usize,
    cr: usize, cc: usize,
    // Pre-computed full-raster images
    sobel: &[f32], lap: &[f32], nir_clean: &[f32],
    // Pre-computed LBP code images (5 bands)
    lbp_nir: &[u8], lbp_ndvi: &[u8], lbp_evi2: &[u8],
    lbp_swir1: &[u8], lbp_ndti: &[u8],
) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let mut fi = 0; // feature index

    // ── 1. Band statistics (80) ──
    let mut band_px = [[0.0f32; N_PX]; N_BANDS];
    for b in 0..N_BANDS {
        let band_off = b * h * w;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for dr in 0..GP {
            let src_off = band_off + (r0 + dr) * w + c0;
            let dst_off = dr * GP;
            band_px[b][dst_off..dst_off + GP]
                .copy_from_slice(&spec[src_off..src_off + GP]);
        }
        let s = cell_stats_8(&band_px[b]);
        for v in s { out[fi] = v; fi += 1; }
    }

    // ── 2. Spectral indices (75) ──
    let mut idx_px = [0.0f32; N_PX];

    // Helper references
    let blue = &band_px[B02]; let green = &band_px[B03]; let red = &band_px[B04];
    let re1 = &band_px[B05]; let re2 = &band_px[B06]; let re3 = &band_px[B07];
    let nir = &band_px[B08]; let swir1 = &band_px[B11]; let swir2 = &band_px[B12];

    // Normalized differences (10 indices)
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
    let mut ndvi_px = [0.0f32; N_PX]; // save for spatial features
    for (pi, &(a, b)) in pairs.iter().enumerate() {
        for i in 0..N_PX {
            idx_px[i] = safe_ratio(band_px[a][i], band_px[b][i]);
        }
        if pi == 0 { ndvi_px = idx_px; } // save NDVI
        let s = cell_stats_5(&idx_px);
        for v in s { out[fi] = v; fi += 1; }
    }

    // SAVI
    for i in 0..N_PX {
        idx_px[i] = if nir[i].is_finite() && red[i].is_finite() {
            1.5 * (nir[i] - red[i]) / (nir[i] + red[i] + 0.5 + EPS)
        } else { f32::NAN };
    }
    let s = cell_stats_5(&idx_px); for v in s { out[fi] = v; fi += 1; }

    // BSI
    for i in 0..N_PX {
        idx_px[i] = if swir1[i].is_finite() && red[i].is_finite()
            && nir[i].is_finite() && blue[i].is_finite() {
            let num = (swir1[i] + red[i]) - (nir[i] + blue[i]);
            num / ((swir1[i] + red[i]) + (nir[i] + blue[i]) + EPS)
        } else { f32::NAN };
    }
    let s = cell_stats_5(&idx_px); for v in s { out[fi] = v; fi += 1; }

    // EVI2
    for i in 0..N_PX {
        idx_px[i] = if nir[i].is_finite() && red[i].is_finite() {
            2.5 * (nir[i] - red[i]) / (nir[i] + 2.4 * red[i] + 1.0 + EPS)
        } else { f32::NAN };
    }
    let s = cell_stats_5(&idx_px); for v in s { out[fi] = v; fi += 1; }

    // IRECI
    for i in 0..N_PX {
        idx_px[i] = if re3[i].is_finite() && red[i].is_finite()
            && re1[i].is_finite() && re2[i].is_finite() {
            (re3[i] - red[i]) / (re1[i] / (re2[i] + EPS) + EPS)
        } else { f32::NAN };
    }
    let s = cell_stats_5(&idx_px); for v in s { out[fi] = v; fi += 1; }

    // CRI1
    for i in 0..N_PX {
        idx_px[i] = if green[i].is_finite() && re1[i].is_finite()
            && green[i] > EPS && re1[i] > EPS {
            (1.0 / green[i]) - (1.0 / re1[i])
        } else { f32::NAN };
    }
    let s = cell_stats_5(&idx_px); for v in s { out[fi] = v; fi += 1; }

    // ── 3. Tasseled Cap (6) ──
    for coeff in &[TC_B, TC_G, TC_W] {
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut n = 0u32;
        for i in 0..N_PX {
            let mut ok = true;
            let mut dot = 0.0f32;
            for b in 0..N_BANDS {
                if !band_px[b][i].is_finite() { ok = false; break; }
                dot += band_px[b][i] * coeff[b];
            }
            if ok { sum += dot as f64; sum2 += (dot as f64) * (dot as f64); n += 1; }
        }
        if n > 0 {
            let mean = sum / n as f64;
            out[fi] = mean as f32;
            out[fi + 1] = ((sum2 / n as f64 - mean * mean).max(0.0)).sqrt() as f32;
        } else {
            out[fi] = f32::NAN; out[fi + 1] = f32::NAN;
        }
        fi += 2;
    }

    // ── 4. Spatial (8) ──
    // Edge stats from pre-computed Sobel
    let e = cell_agg_3(sobel, w, cr, cc);
    out[fi] = e[0]; fi += 1; out[fi] = e[1]; fi += 1; out[fi] = e[2]; fi += 1;

    // Laplacian stats from pre-computed Laplacian
    let l = cell_lap_stats(lap, w, cr, cc);
    out[fi] = l[0]; fi += 1; out[fi] = l[1]; fi += 1;

    // Moran's I (from clean NIR)
    let nir_px = extract_cell(nir_clean, w, cr, cc);
    out[fi] = cell_morans_i(&nir_px); fi += 1;

    // NDVI range & IQR
    let ndvi_s = cell_stats_8(&ndvi_px);
    out[fi] = ndvi_s[3] - ndvi_s[2]; fi += 1; // max - min
    out[fi] = ndvi_s[6] - ndvi_s[4]; fi += 1; // q75 - q25

    // ── 5. Multi-band LBP (55) ──
    let lbp_imgs = [lbp_nir, lbp_ndvi, lbp_evi2, lbp_swir1, lbp_ndti];
    for lbp in &lbp_imgs {
        let h = cell_lbp_hist(lbp, w, cr, cc);
        for i in 0..LBP_BINS + 1 { out[fi] = h[i]; fi += 1; }
    }

    debug_assert_eq!(fi, N_FEAT);
    out
}

// =====================================================================
// Python interface
// =====================================================================

/// Extract all features for one season.
///
/// spectral: (10, H, W) f32, already normalised (divided by scale)
/// valid_frac: (H, W) f32
/// Returns: (n_cells * 224,) f32 flat array
#[pyfunction]
fn extract_season<'py>(
    py: Python<'py>,
    spectral: PyReadonlyArray3<'py, f32>,
    valid_frac: PyReadonlyArray2<'py, f32>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let spec_view = spectral.as_array();
    let _vf = valid_frac.as_array();

    let h = spec_view.shape()[1];
    let w = spec_view.shape()[2];
    assert_eq!(spec_view.shape()[0], N_BANDS);
    assert_eq!(h, n_rows * GP);
    assert_eq!(w, n_cols * GP);

    // Zero-copy if contiguous, otherwise copy
    let spec: Vec<f32>;
    let spec_slice: &[f32] = match spec_view.as_slice() {
        Some(s) => s,
        None => {
            spec = spec_view.iter().copied().collect();
            &spec
        }
    };

    let n_cells = n_rows * n_cols;
    let lbp_lut = build_lbp_lut();

    // ── Phase 1: Full-raster pre-computation ──

    // Extract normalised band slices
    let band_slice = |b: usize| -> &[f32] { &spec_slice[b * h * w..(b + 1) * h * w] };

    // Clean NIR (NaN-filled, for Sobel/Moran's)
    let nir_clean = clean_band(band_slice(B08), h, w);

    // Sobel magnitude on clean NIR (full raster, row-parallel)
    let sobel = compute_sobel_mag(&nir_clean, h, w);

    // Laplacian on clean NIR
    let laplacian = compute_laplacian(&nir_clean, h, w);

    // LBP band images (NaN-filled, scaled to 0-1)
    let red_clean = clean_band(band_slice(B04), h, w);
    let swir1_clean = clean_band(band_slice(B11), h, w);
    let swir2_clean = clean_band(band_slice(B12), h, w);

    // NDVI image [0, 1]
    let ndvi_img: Vec<f32> = (0..h * w).map(|i| {
        let ndvi = (nir_clean[i] - red_clean[i]) / (nir_clean[i] + red_clean[i] + EPS);
        ((ndvi + 1.0) / 2.0).clamp(0.0, 1.0)
    }).collect();

    // EVI2 image [0, 1]
    let evi2_img: Vec<f32> = (0..h * w).map(|i| {
        let e = 2.5 * (nir_clean[i] - red_clean[i])
            / (nir_clean[i] + 2.4 * red_clean[i] + 1.0 + EPS);
        ((e + 0.5) / 1.5).clamp(0.0, 1.0)
    }).collect();

    // NDTI image [0, 1]
    let ndti_img: Vec<f32> = (0..h * w).map(|i| {
        let n = (swir1_clean[i] - swir2_clean[i]) / (swir1_clean[i] + swir2_clean[i] + EPS);
        ((n + 1.0) / 2.0).clamp(0.0, 1.0)
    }).collect();

    // Full-raster LBP for 5 bands (row-parallel each)
    let lbp_nir = compute_lbp_raster(&nir_clean, h, w, &lbp_lut);
    let lbp_ndvi = compute_lbp_raster(&ndvi_img, h, w, &lbp_lut);
    let lbp_evi2 = compute_lbp_raster(&evi2_img, h, w, &lbp_lut);
    let lbp_swir1 = compute_lbp_raster(&swir1_clean, h, w, &lbp_lut);
    let lbp_ndti = compute_lbp_raster(&ndti_img, h, w, &lbp_lut);

    // ── Phase 2: Per-cell extraction (cell-parallel) ──
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

    // Flatten
    let mut flat = Vec::with_capacity(n_cells * N_FEAT);
    for feats in &results {
        flat.extend_from_slice(feats);
    }

    Ok(ndarray::Array1::from_vec(flat).into_pyarray(py).into())
}

#[pyfunction]
fn n_features_per_cell() -> usize { N_FEAT }

#[pyfunction]
fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FEAT);
    let bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"];
    let bst = ["mean","std","min","max","q25","median","q75","finite_frac"];
    for bn in &bands { for sn in &bst { names.push(format!("{bn}_{sn}")); } }

    let idxs = ["NDVI","NDWI","NDBI","NDMI","NBR","NDRE1","NDRE2",
                "MNDWI","GNDVI","NDTI","SAVI","BSI","EVI2","IRECI","CRI1"];
    let ist = ["mean","std","q25","median","q75"];
    for idn in &idxs { for sn in &ist { names.push(format!("{idn}_{sn}")); } }

    for tc in &["TC_bright","TC_green","TC_wet"] {
        names.push(format!("{tc}_mean")); names.push(format!("{tc}_std"));
    }
    names.extend(["edge_mean","edge_std","edge_max","lap_abs_mean","lap_std",
                   "morans_I_NIR","NDVI_range","NDVI_iqr"].iter().map(|s| s.to_string()));

    for lb in &["NIR","NDVI","EVI2","SWIR1","NDTI"] {
        for b in 0..LBP_BINS { names.push(format!("LBP_{lb}_u{LBP_P}_{b}")); }
        names.push(format!("LBP_{lb}_entropy"));
    }
    assert_eq!(names.len(), N_FEAT);
    names
}

#[pymodule]
fn terrapulse_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_season, m)?)?;
    m.add_function(wrap_pyfunction!(n_features_per_cell, m)?)?;
    m.add_function(wrap_pyfunction!(feature_names, m)?)?;
    Ok(())
}
