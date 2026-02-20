//! SAR (Sentinel-1) feature extraction: VV/VH statistics, indices, and LBP texture.
//! 48 features per cell per season.

use rayon::prelude::*;

use crate::features::{
    build_lbp_lut, cell_lbp_hist, cell_stats_5, cell_stats_8, compute_lbp_perpatch, EPS, GP,
    LBP_BINS,
};

const N_PX: usize = GP * GP; // 100 pixels per cell

// SAR band indices within 2-band TIF
const SAR_VV: usize = 0;
const SAR_VH: usize = 1;
pub const N_SAR_BANDS: usize = 2;

// Feature counts per season:
//   VV stats: 8, VH stats: 8
//   CR (VH/VV) stats: 5, RVI stats: 5
//   LBP(VV): 11, LBP(VH): 11
//   Total: 48
const N_SAR_BAND_STATS: usize = N_SAR_BANDS * 8; // 16
const N_SAR_IDX_STATS: usize = 2 * 5; // 10 (CR + RVI, 5 stats each)
const N_SAR_LBP: usize = 2 * (LBP_BINS + 1); // 22 (VV + VH, 11 each)
pub const N_SAR_FEAT: usize = N_SAR_BAND_STATS + N_SAR_IDX_STATS + N_SAR_LBP; // 48

/// Compute cross-polarization ratio = VH / VV
#[inline(always)]
fn sar_cross_ratio(vv: f32, vh: f32) -> f32 {
    if vv.is_finite() && vh.is_finite() && vv.abs() > EPS {
        vh / vv
    } else {
        f32::NAN
    }
}

/// Compute Radar Vegetation Index = 4 * VH / (VV + VH)
#[inline(always)]
fn sar_rvi(vv: f32, vh: f32) -> f32 {
    if vv.is_finite() && vh.is_finite() {
        let denom = vv + vh;
        if denom.abs() > EPS {
            4.0 * vh / denom
        } else {
            f32::NAN
        }
    } else {
        f32::NAN
    }
}

/// Extract 48 SAR features for one cell.
fn extract_sar_cell_features(
    sar_data: &[f32],
    h: usize,
    w: usize,
    cr: usize,
    cc: usize,
    lbp_vv: &[u8],
    lbp_vh: &[u8],
) -> [f32; N_SAR_FEAT] {
    let mut out = [0.0f32; N_SAR_FEAT];
    let mut fi: usize = 0;

    // 1) Band stats (16: 8 for VV, 8 for VH)
    let mut band_px = [[0.0f32; N_PX]; N_SAR_BANDS];
    for b in 0..N_SAR_BANDS {
        let band_off = b * h * w;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for dr in 0..GP {
            let src_off = band_off + (r0 + dr) * w + c0;
            let dst_off = dr * GP;
            band_px[b][dst_off..dst_off + GP].copy_from_slice(&sar_data[src_off..src_off + GP]);
        }
        let s = cell_stats_8(&band_px[b]);
        for v in s {
            out[fi] = v;
            fi += 1;
        }
    }

    // 2) SAR indices (10: 5 for CR, 5 for RVI)
    let vv = &band_px[SAR_VV];
    let vh = &band_px[SAR_VH];

    // Cross-pol ratio
    let mut idx_px = [0.0f32; N_PX];
    for i in 0..N_PX {
        idx_px[i] = sar_cross_ratio(vv[i], vh[i]);
    }
    let s = cell_stats_5(&idx_px);
    for v in s {
        out[fi] = v;
        fi += 1;
    }

    // RVI
    for i in 0..N_PX {
        idx_px[i] = sar_rvi(vv[i], vh[i]);
    }
    let s = cell_stats_5(&idx_px);
    for v in s {
        out[fi] = v;
        fi += 1;
    }

    // 3) LBP texture (22: 11 for VV, 11 for VH)
    let lbp_vv_hist = cell_lbp_hist(lbp_vv, w, cr, cc);
    for v in lbp_vv_hist {
        out[fi] = v;
        fi += 1;
    }

    let lbp_vh_hist = cell_lbp_hist(lbp_vh, w, cr, cc);
    for v in lbp_vh_hist {
        out[fi] = v;
        fi += 1;
    }

    debug_assert_eq!(fi, N_SAR_FEAT);
    out
}

/// Column names for SAR features (48 names).
pub fn sar_feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_SAR_FEAT);

    // Band stats
    let bands = ["SAR_VV", "SAR_VH"];
    let bst = [
        "mean",
        "std",
        "min",
        "max",
        "q25",
        "median",
        "q75",
        "finite_frac",
    ];
    for bn in &bands {
        for sn in &bst {
            names.push(format!("{bn}_{sn}"));
        }
    }

    // Index stats
    let idxs = ["SAR_CR", "SAR_RVI"];
    let ist = ["mean", "std", "q25", "median", "q75"];
    for idn in &idxs {
        for sn in &ist {
            names.push(format!("{idn}_{sn}"));
        }
    }

    // LBP
    for lb in &["SAR_LBP_VV", "SAR_LBP_VH"] {
        for b in 0..LBP_BINS {
            names.push(format!("{lb}_u8_{b}"));
        }
        names.push(format!("{lb}_entropy"));
    }

    assert_eq!(names.len(), N_SAR_FEAT);
    names
}

/// Extract SAR features for all seasons. Returns [n_cells, n_seasons * N_SAR_FEAT] flat vector.
///
/// `season_data`: Vec of flat f32 arrays, each [N_SAR_BANDS * H * W].
/// `n_rows`, `n_cols`: grid dimensions (H = n_rows * GP, W = n_cols * GP).
pub fn extract_all_sar_seasons(season_data: &[Vec<f32>], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let h = n_rows * GP;
    let w = n_cols * GP;
    let n_seasons = season_data.len();
    let n_cells = n_rows * n_cols;
    let total_feats = n_cells * n_seasons * N_SAR_FEAT;

    let lbp_lut = build_lbp_lut();

    let season_results: Vec<Vec<[f32; N_SAR_FEAT]>> = season_data
        .iter()
        .map(|sar_slice| {
            let band_slice = |b: usize| -> &[f32] { &sar_slice[b * h * w..(b + 1) * h * w] };

            // LBP on VV and VH (per-patch, with clipping to [0,1] since SAR is
            // already normalized to [0,1] after dB conversion + scaling)
            let lbp_vv =
                compute_lbp_perpatch(band_slice(SAR_VV), h, w, n_rows, n_cols, &lbp_lut, true);
            let lbp_vh =
                compute_lbp_perpatch(band_slice(SAR_VH), h, w, n_rows, n_cols, &lbp_lut, true);

            (0..n_cells)
                .into_par_iter()
                .map(|ci| {
                    extract_sar_cell_features(
                        sar_slice,
                        h,
                        w,
                        ci / n_cols,
                        ci % n_cols,
                        &lbp_vv,
                        &lbp_vh,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Interleave: for each cell, concatenate all seasons' SAR features
    let mut flat = vec![0.0f32; total_feats];
    for ci in 0..n_cells {
        let cell_base = ci * n_seasons * N_SAR_FEAT;
        for (si, season) in season_results.iter().enumerate().take(n_seasons) {
            let dst = cell_base + si * N_SAR_FEAT;
            flat[dst..dst + N_SAR_FEAT].copy_from_slice(&season[ci]);
        }
    }

    flat
}
