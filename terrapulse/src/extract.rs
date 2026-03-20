//! Feature extraction module: reads seasonal GeoTIFFs, runs feature extraction,
//! writes output as parquet.

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::features;
use crate::parquet_io;
use crate::sar_features;
use crate::tif_reader;

const SEASONS: [&str; 3] = ["spring", "summer", "autumn"];
const NODATA: f32 = -9999.0;

/// Detect if data is in DN (> 1.0 scale) and compute scale factor.
fn detect_scale(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    let step = (data.len() / 10000).max(1);
    for &v in data.iter().step_by(step) {
        if v.is_finite() && v > 0.0 && v != NODATA {
            sum += v as f64;
            n += 1;
        }
    }
    // Fallback if still no valid pixels found
    if n == 0 {
        for &v in data {
            if v.is_finite() && v > 0.0 && v != NODATA {
                sum += v as f64;
                n += 1;
                if n > 1000 {
                    break;
                }
            }
        }
    }
    if n == 0 {
        return 1.0;
    }
    let mean = sum / n as f64;
    if mean > 100.0 {
        10000.0
    } else if mean > 10.0 {
        1000.0
    } else {
        1.0
    }
}

/// Extract features for a year pair and write to parquet.
pub fn extract_year_pair(
    prev_year: u32,
    curr_year: u32,
    region_name: &str,
    raw_dir: &Path,
    features_dir: &Path,
    min_valid_frac: f32,
) -> Result<Option<PathBuf>> {
    let tag = format!("{prev_year}_{curr_year}");
    let out_path = features_dir.join(format!("features_rust_{tag}.parquet"));

    if out_path.exists() {
        println!("  [{tag}] Already extracted -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(features_dir)?;

    // Year mapping (always use 2020/2021 model tags)
    let year_map = [(prev_year, 2020u32), (curr_year, 2021)];

    // Check all TIFs exist
    let mut jobs = Vec::new();
    for &(actual_year, _model_year) in &year_map {
        for season in SEASONS {
            let tif = raw_dir.join(format!(
                "sentinel2_{region_name}_{actual_year}_{season}.tif"
            ));
            if !tif.exists() {
                println!("  [{tag}] WARNING: Missing {} -- skip", tif.display());
                return Ok(None);
            }
            jobs.push((actual_year, season));
        }
    }

    let t0 = std::time::Instant::now();

    // Load all seasonal rasters using native TIF reader
    let mut spectral_list: Vec<Vec<f32>> = Vec::new();
    let mut suffixes = Vec::new();
    let mut nr = 0usize;
    let mut nc = 0usize;
    let mut vf_min: Option<Vec<f32>> = None;

    for (actual_year, season) in jobs.iter() {
        let model_year = year_map.iter().find(|(a, _)| a == actual_year).unwrap().1;
        let tif = raw_dir.join(format!(
            "sentinel2_{region_name}_{actual_year}_{season}.tif"
        ));

        let t_read = std::time::Instant::now();

        // Always read bands + valid fraction together (single decode)
        let (nb, h, w, mut data, vf) =
            tif_reader::read_tif_bands_and_valid_fraction(&tif, features::N_BANDS)?;

        if nr == 0 {
            nr = h / features::GP;
            nc = w / features::GP;
        }

        // Accumulate minimum valid fraction across all rasters
        if let Some(vf_data) = vf {
            vf_min = Some(match vf_min {
                None => vf_data,
                Some(prev) => prev
                    .iter()
                    .zip(vf_data.iter())
                    .map(|(&a, &b)| match (a.is_finite(), b.is_finite()) {
                        (true, true) => a.min(b),
                        (true, false) => a,
                        (false, true) => b,
                        (false, false) => f32::NAN,
                    })
                    .collect(),
            });
        }

        let read_ms = t_read.elapsed().as_millis();
        assert!(
            nb >= features::N_BANDS,
            "TIF has {nb} bands, need {}",
            features::N_BANDS
        );

        // Normalize to [0,1] if in DN scale
        let scale = detect_scale(&data);
        if scale != 1.0 {
            for v in data.iter_mut() {
                if v.is_finite() && *v != NODATA {
                    *v /= scale;
                }
            }
        }
        // Replace NODATA with NaN
        for v in data.iter_mut() {
            if *v == NODATA {
                *v = f32::NAN;
            }
        }

        spectral_list.push(data);
        suffixes.push(format!("{model_year}_{season}"));
        println!("    Loaded {actual_year}_{season} -> {model_year}_{season} ({read_ms}ms)");
    }

    let n_cells = nr * nc;

    // Run extraction
    let t1 = std::time::Instant::now();
    let flat = features::extract_all_seasons(&spectral_list, nr, nc);
    let dt = t1.elapsed().as_secs_f64();
    println!("    Rust extraction: {dt:.1}s for {} seasons", jobs.len());
    drop(spectral_list);

    let n_seasons = suffixes.len();

    // Build column names for per-season features
    let base_names = features::feature_names();
    let mut columns = Vec::with_capacity(n_seasons * features::N_FEAT + 200);
    for suffix in &suffixes {
        for name in &base_names {
            columns.push(format!("{name}_{suffix}"));
        }
    }

    // Build row data [n_cells][n_features_total] from optical features
    let n_total_feats = n_seasons * features::N_FEAT;
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n_cells);
    for ci in 0..n_cells {
        let base = ci * n_total_feats;
        let mut row: Vec<f32> = flat[base..base + n_total_feats].to_vec();

        // Replace inf with NaN, then impute NaN with column medians later
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = f32::NAN;
            }
        }
        rows.push(row);
    }

    // =========================================================================
    // SAR feature extraction (optional — backward-compatible)
    // =========================================================================
    let mut has_sar = true;
    let mut sar_spectral_list: Vec<Vec<f32>> = Vec::new();

    for (actual_year, season) in jobs.iter() {
        let sar_tif = raw_dir.join(format!(
            "sentinel1_{region_name}_{actual_year}_{season}.tif"
        ));
        if !sar_tif.exists() {
            has_sar = false;
            break;
        }
    }

    if has_sar {
        println!("    SAR TIFs detected — extracting SAR features");
        for (actual_year, season) in jobs.iter() {
            let sar_tif = raw_dir.join(format!(
                "sentinel1_{region_name}_{actual_year}_{season}.tif"
            ));
            let t_read = std::time::Instant::now();

            let (nb, _h, _w, mut data, _vf) =
                tif_reader::read_tif_bands_and_valid_fraction(&sar_tif, sar_features::N_SAR_BANDS)?;

            assert!(
                nb >= sar_features::N_SAR_BANDS,
                "SAR TIF has {nb} bands, need {}",
                sar_features::N_SAR_BANDS
            );

            // Replace NODATA with NaN
            for v in data.iter_mut() {
                if *v == NODATA {
                    *v = f32::NAN;
                }
            }

            // SAR data should already be in [0,1] after dB conversion + scaling in Python.
            // But if raw linear power values are present, detect and convert.
            let mut finite_sum = 0.0f64;
            let mut finite_n = 0u64;
            let step = (data.len() / 10000).max(1);
            for &v in data.iter().step_by(step) {
                if v.is_finite() && v > 0.0 && v != NODATA {
                    finite_sum += v as f64;
                    finite_n += 1;
                }
            }
            if finite_n == 0 {
                for &v in data.iter() {
                    if v.is_finite() && v > 0.0 && v != NODATA {
                        finite_sum += v as f64;
                        finite_n += 1;
                        if finite_n > 1000 {
                            break;
                        }
                    }
                }
            }
            if finite_n > 0 {
                let mean_val = finite_sum / finite_n as f64;
                if mean_val > 1.5 {
                    // Likely raw linear power, convert to dB then scale to [0,1]
                    println!(
                        "      SAR: converting from linear power (mean={mean_val:.3}) to [0,1]"
                    );
                    for v in data.iter_mut() {
                        if v.is_finite() && *v > 0.0 {
                            let db = 10.0 * v.log10();
                            // Clamp to [-30, 0] and scale to [0, 1]
                            *v = (db.clamp(-30.0, 0.0) + 30.0) / 30.0;
                        } else if v.is_finite() {
                            *v = 0.0; // zero or negative power → 0
                        }
                    }
                }
            }

            let read_ms = t_read.elapsed().as_millis();
            println!("      Loaded SAR {actual_year}_{season} ({read_ms}ms)");
            sar_spectral_list.push(data);
        }

        // Run SAR extraction
        let t1 = std::time::Instant::now();
        let sar_flat = sar_features::extract_all_sar_seasons(&sar_spectral_list, nr, nc);
        let dt = t1.elapsed().as_secs_f64();
        println!(
            "    SAR extraction: {dt:.1}s for {} seasons",
            sar_spectral_list.len()
        );
        drop(sar_spectral_list);

        // Add SAR column names
        let sar_base_names = sar_features::sar_feature_names();
        for suffix in &suffixes {
            for name in &sar_base_names {
                columns.push(format!("{name}_{suffix}"));
            }
        }

        // Append SAR features to each row
        let n_sar_total = n_seasons * sar_features::N_SAR_FEAT;
        for ci in 0..n_cells {
            let base = ci * n_sar_total;
            let sar_row: Vec<f32> = sar_flat[base..base + n_sar_total]
                .iter()
                .map(|&v| if v.is_finite() { v } else { f32::NAN })
                .collect();
            rows[ci].extend_from_slice(&sar_row);
        }

        println!("    Added {} SAR columns per cell", n_sar_total);
    } else {
        println!("    No SAR TIFs found — optical-only mode (backward-compatible)");
    }

    // =========================================================================
    // Cell-level NaN fill: temporal (same season, other year) + spatial NN
    // Applied BEFORE phenological features so pheno gets clean inputs.
    // =========================================================================
    {
        let n_optical = n_seasons * features::N_FEAT;
        let n_sar_per_season = if has_sar { sar_features::N_SAR_FEAT } else { 0 };
        let n_row_len = rows.first().map_or(0, |r| r.len());

        let mut temporal_fills = 0u64;

        // --- Step 1: Temporal fill ---
        // For each feature in each season, if NaN, try same feature from same
        // season but other year. E.g., autumn_2020_NDVI_mean -> autumn_2021_NDVI_mean
        // Seasons come in groups of 3 per year: [spring0, summer0, autumn0, spring1, summer1, autumn1]
        let n_years_loaded = n_seasons / 3;
        if n_years_loaded == 2 {
            // Map season index pairs: 0<->3 (spring), 1<->4 (summer), 2<->5 (autumn)
            for row in rows.iter_mut() {
                for season_in_year in 0..3 {
                    let si_a = season_in_year;         // year 0
                    let si_b = 3 + season_in_year;     // year 1

                    // Optical features
                    for fi in 0..features::N_FEAT {
                        let idx_a = si_a * features::N_FEAT + fi;
                        let idx_b = si_b * features::N_FEAT + fi;
                        if !row[idx_a].is_finite() && row[idx_b].is_finite() {
                            row[idx_a] = row[idx_b];
                            temporal_fills += 1;
                        } else if row[idx_a].is_finite() && !row[idx_b].is_finite() {
                            row[idx_b] = row[idx_a];
                            temporal_fills += 1;
                        }
                    }

                    // SAR features
                    if has_sar {
                        for fi in 0..n_sar_per_season {
                            let idx_a = n_optical + si_a * n_sar_per_season + fi;
                            let idx_b = n_optical + si_b * n_sar_per_season + fi;
                            if idx_a < n_row_len && idx_b < n_row_len {
                                if !row[idx_a].is_finite() && row[idx_b].is_finite() {
                                    row[idx_a] = row[idx_b];
                                    temporal_fills += 1;
                                } else if row[idx_a].is_finite() && !row[idx_b].is_finite() {
                                    row[idx_b] = row[idx_a];
                                    temporal_fills += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // NOTE: Spatial NN fill (cross-cell) was removed — it leaked land
        // features into water/coastal cells. Remaining NaN will be zero-filled
        // in the final pass below.
        let remaining_nan = rows
            .iter()
            .flat_map(|r| r.iter())
            .filter(|v| !v.is_finite())
            .count();
        println!("    NaN fill: {temporal_fills} temporal, {remaining_nan} still missing before final zero-fill");
    }

    // =========================================================================
    // Phenological cross-season features (V4)
    // For each year's 3 seasons (spring, summer, autumn), compute:
    //   curvature = summer - (spring + autumn) / 2  (seasonal peak)
    //   slope     = (autumn - spring) / 2           (greening/browning trend)
    //   amplitude = max - min across 3 seasons       (seasonal variability)
    //   peak      = argmax(spring, summer, autumn)   (timing of peak, 0/1/2)
    // Applied to 10 band means + 5 key index means = 15 signals per year
    // =========================================================================
    if n_seasons >= 3 {
        // Offsets of mean values within N_FEAT for each signal
        // Bands: mean is at offset 0, 8, 16, ..., 72 (10 bands × 8 stats, mean is first)
        // Indices: mean is at offset 80, 85, 90, ..., 150 (15 indices × 5 stats, mean is first)
        // We use 10 bands + 5 key indices (NDVI, NDWI, NDBI, BSI, EVI2)
        let band_mean_offsets: Vec<usize> = (0..10).map(|b| b * 8).collect(); // 0,8,16,...,72
                                                                              // NDVI=0, NDWI=1, NDBI=2, BSI=11, EVI2=12 within the 15 indices
        let idx_mean_offsets: Vec<usize> = vec![
            80 + 0 * 5,  // NDVI mean
            80 + 1 * 5,  // NDWI mean
            80 + 2 * 5,  // NDBI mean
            80 + 11 * 5, // BSI mean
            80 + 12 * 5, // EVI2 mean
        ];

        let all_offsets: Vec<usize> = band_mean_offsets
            .iter()
            .chain(idx_mean_offsets.iter())
            .copied()
            .collect();

        let signal_names = [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI",
            "NDBI", "BSI", "EVI2",
        ];

        let pheno_names = ["curvature", "slope", "amplitude", "peak"];

        // Process each year separately (seasons come in groups of 3)
        let n_years = n_seasons / 3;
        for yr_idx in 0..n_years {
            let spring_season = yr_idx * 3; // index 0 or 3
            let summer_season = yr_idx * 3 + 1; // index 1 or 4
            let autumn_season = yr_idx * 3 + 2; // index 2 or 5

            let year_tag = &suffixes[spring_season]; // e.g. "2020_spring"
            let year_label = year_tag.split('_').next().unwrap_or("unknown");

            // Add column names for this year's phenological features
            for sig_name in &signal_names {
                for pheno in &pheno_names {
                    columns.push(format!("{sig_name}_pheno_{pheno}_{year_label}"));
                }
            }

            // Compute phenological features for each cell
            for row in rows.iter_mut() {
                for &offset in &all_offsets {
                    let spring_val = row[spring_season * features::N_FEAT + offset];
                    let summer_val = row[summer_season * features::N_FEAT + offset];
                    let autumn_val = row[autumn_season * features::N_FEAT + offset];

                    // curvature: summer peak relative to shoulders
                    let curvature = summer_val - (spring_val + autumn_val) / 2.0;
                    // slope: trend from spring to autumn
                    let slope = (autumn_val - spring_val) / 2.0;
                    // amplitude: max - min
                    let mx = spring_val.max(summer_val).max(autumn_val);
                    let mn = spring_val.min(summer_val).min(autumn_val);
                    let amplitude = mx - mn;
                    // peak_season: 0=spring, 1=summer, 2=autumn
                    let peak = if summer_val >= spring_val && summer_val >= autumn_val {
                        1.0f32
                    } else if autumn_val >= spring_val {
                        2.0f32
                    } else {
                        0.0f32
                    };

                    row.push(curvature);
                    row.push(slope);
                    row.push(amplitude);
                    row.push(peak);
                }
            }
        }

        let n_pheno = all_offsets.len() * 4 * n_years;
        println!("    Added {n_pheno} optical phenological features ({} signals x 4 pheno x {n_years} years)", all_offsets.len());

        // =====================================================================
        // SAR Phenological cross-season features
        // For each year's 3 SAR seasons, compute curvature/slope/amplitude/peak
        // Applied to: VV_mean, VH_mean, CR_mean, RVI_mean = 4 signals per year
        // =====================================================================
        if has_sar {
            // SAR feature offsets within N_SAR_FEAT (48):
            // VV_mean = offset 0, VH_mean = offset 8
            // CR_mean = offset 16, RVI_mean = offset 21
            let sar_mean_offsets: Vec<usize> = vec![0, 8, 16, 21];
            let sar_signal_names = ["SAR_VV", "SAR_VH", "SAR_CR", "SAR_RVI"];
            let pheno_names_sar = ["curvature", "slope", "amplitude", "peak"];

            // SAR features start after optical features in each row
            let optical_per_season = features::N_FEAT;
            let sar_per_season = sar_features::N_SAR_FEAT;
            // In the row layout:
            // [optical_season_0..optical_season_N | sar_season_0..sar_season_N | ...]
            // Optical: n_seasons * N_FEAT columns from index 0
            // SAR: n_seasons * N_SAR_FEAT columns from index (n_seasons * N_FEAT)
            let sar_base_offset = n_seasons * optical_per_season;

            for yr_idx in 0..n_years {
                let spring_season = yr_idx * 3;
                let summer_season = yr_idx * 3 + 1;
                let autumn_season = yr_idx * 3 + 2;

                let year_tag = &suffixes[spring_season];
                let year_label = year_tag.split('_').next().unwrap_or("unknown");

                for sig_name in &sar_signal_names {
                    for pheno in &pheno_names_sar {
                        columns.push(format!("{sig_name}_pheno_{pheno}_{year_label}"));
                    }
                }

                for row in rows.iter_mut() {
                    for &offset in &sar_mean_offsets {
                        let spring_val =
                            row[sar_base_offset + spring_season * sar_per_season + offset];
                        let summer_val =
                            row[sar_base_offset + summer_season * sar_per_season + offset];
                        let autumn_val =
                            row[sar_base_offset + autumn_season * sar_per_season + offset];

                        let curvature = summer_val - (spring_val + autumn_val) / 2.0;
                        let slope = (autumn_val - spring_val) / 2.0;
                        let mx = spring_val.max(summer_val).max(autumn_val);
                        let mn = spring_val.min(summer_val).min(autumn_val);
                        let amplitude = mx - mn;
                        let peak = if summer_val >= spring_val && summer_val >= autumn_val {
                            1.0f32
                        } else if autumn_val >= spring_val {
                            2.0f32
                        } else {
                            0.0f32
                        };

                        row.push(curvature);
                        row.push(slope);
                        row.push(amplitude);
                        row.push(peak);
                    }
                }
            }

            let n_sar_pheno = sar_mean_offsets.len() * 4 * n_years;
            println!("    Added {n_sar_pheno} SAR phenological features ({} signals x 4 pheno x {n_years} years)", sar_mean_offsets.len());

            // =================================================================
            // SAR Temporal Features (new cross-season statistics)
            // 8 features per year:
            //   3 summer-winter contrasts: VH, VV, CR (spring as winter proxy)
            //   3 temporal_std: std(mean across 3 seasons) for VH, VV, CR
            //   2 temporal_cv: temporal_std / temporal_mean for VH, VV
            // =================================================================
            // Offsets for VV_mean, VH_mean, CR_mean within N_SAR_FEAT
            let temporal_offsets: Vec<usize> = vec![0, 8, 16]; // VV, VH, CR
            let temporal_names = ["SAR_VV", "SAR_VH", "SAR_CR"];

            for yr_idx in 0..n_years {
                let spring_season = yr_idx * 3;
                let summer_season = yr_idx * 3 + 1;
                let _autumn_season = yr_idx * 3 + 2;

                let year_tag = &suffixes[spring_season];
                let year_label = year_tag.split('_').next().unwrap_or("unknown");

                // Column names: summer_winter contrasts
                for sig in &temporal_names {
                    columns.push(format!("{sig}_summer_winter_{year_label}"));
                }
                // Column names: temporal_std
                for sig in &temporal_names {
                    columns.push(format!("{sig}_temporal_std_{year_label}"));
                }
                // Column names: temporal_cv (VV and VH only, not CR)
                columns.push(format!("SAR_VV_temporal_cv_{year_label}"));
                columns.push(format!("SAR_VH_temporal_cv_{year_label}"));

                // Compute for each cell
                for row in rows.iter_mut() {
                    // Summer-winter contrasts (spring as winter proxy)
                    for &offset in &temporal_offsets {
                        let spring_val =
                            row[sar_base_offset + spring_season * sar_per_season + offset];
                        let summer_val =
                            row[sar_base_offset + summer_season * sar_per_season + offset];
                        row.push(summer_val - spring_val);
                    }

                    // Temporal std across 3 seasons
                    for &offset in &temporal_offsets {
                        let s0 = row[sar_base_offset + (yr_idx * 3) * sar_per_season + offset];
                        let s1 = row[sar_base_offset + (yr_idx * 3 + 1) * sar_per_season + offset];
                        let s2 = row[sar_base_offset + (yr_idx * 3 + 2) * sar_per_season + offset];
                        let mean = (s0 + s1 + s2) / 3.0;
                        let var =
                            ((s0 - mean).powi(2) + (s1 - mean).powi(2) + (s2 - mean).powi(2)) / 3.0;
                        row.push(var.max(0.0).sqrt());
                    }

                    // Temporal CV for VV and VH only (offsets 0 and 8)
                    for &offset in &[0usize, 8usize] {
                        let s0 = row[sar_base_offset + (yr_idx * 3) * sar_per_season + offset];
                        let s1 = row[sar_base_offset + (yr_idx * 3 + 1) * sar_per_season + offset];
                        let s2 = row[sar_base_offset + (yr_idx * 3 + 2) * sar_per_season + offset];
                        let mean = (s0 + s1 + s2) / 3.0;
                        let var =
                            ((s0 - mean).powi(2) + (s1 - mean).powi(2) + (s2 - mean).powi(2)) / 3.0;
                        let std = var.max(0.0).sqrt();
                        let cv = if mean.abs() > 1e-10 {
                            std / mean.abs()
                        } else {
                            0.0
                        };
                        row.push(cv);
                    }
                }
            }

            let n_sar_temporal = 8 * n_years;
            println!("    Added {n_sar_temporal} SAR temporal features (8 x {n_years} years)");
        }
    }

    // Compute per-cell NaN fraction BEFORE final zero-fill.
    // This tracks data quality: 0.0 = all features valid, 1.0 = all features were NaN.
    let n_all_cols = rows.first().map_or(0, |r| r.len());
    let mut nan_fractions: Vec<f32> = Vec::with_capacity(n_cells);
    for row in rows.iter() {
        let nan_count = row.iter().filter(|v| !v.is_finite()).count();
        let frac = if n_all_cols > 0 {
            nan_count as f32 / n_all_cols as f32
        } else {
            0.0
        };
        nan_fractions.push(frac);
    }

    let nan_cells_above_50 = nan_fractions.iter().filter(|&&f| f > 0.5).count();
    let nan_cells_above_0 = nan_fractions.iter().filter(|&&f| f > 0.0).count();
    println!(
        "    Data quality: {}/{} cells fully valid, {} partially missing, {} >50% missing (nodata)",
        n_cells - nan_cells_above_0,
        n_cells,
        nan_cells_above_0 - nan_cells_above_50,
        nan_cells_above_50,
    );

    // Final NaN safety net: any remaining NaN (e.g. from pheno features
    // where all 3 seasons were still NaN after fill) gets zero-filled.
    let mut final_nan_count = 0u64;
    for row in rows.iter_mut() {
        for col in 0..n_all_cols.min(row.len()) {
            if !row[col].is_finite() {
                row[col] = 0.0;
                final_nan_count += 1;
            }
        }
    }
    if final_nan_count > 0 {
        println!("    Final zero-fill for {final_nan_count} remaining NaN values");
    }

    // Add cell_id, valid_fraction, nan_fraction columns
    let mut extra_cols = vec!["cell_id".to_string()];
    let mut extra_data: Vec<Vec<f32>> = vec![(0..n_cells as u32).map(|i| i as f32).collect()];

    // Always add nan_fraction
    extra_cols.push("nan_fraction".to_string());
    extra_data.push(nan_fractions);

    if let Some(ref vf) = vf_min {
        // Aggregate valid fraction per cell (mean of GP×GP pixels)
        let gp = features::GP;
        let mut vf_cells = vec![0.0f32; n_cells];
        for ci in 0..n_cells {
            let cr = ci / nc;
            let cc = ci % nc;
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for dr in 0..gp {
                let r = cr * gp + dr;
                for dc in 0..gp {
                    let c = cc * gp + dc;
                    let v = vf[r * (nc * gp) + c];
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            vf_cells[ci] = if n > 0 { sum / n as f32 } else { 0.0 };
        }
        extra_cols.push("valid_fraction".to_string());
        extra_cols.push("low_valid_fraction".to_string());
        let low_vf: Vec<f32> = vf_cells
            .iter()
            .map(|&v| if v < min_valid_frac { 1.0 } else { 0.0 })
            .collect();
        extra_data.push(vf_cells);
        extra_data.push(low_vf);
    }

    // Write parquet
    parquet_io::write_feature_parquet(&out_path, &extra_cols, &extra_data, &columns, &rows)?;

    let elapsed = t0.elapsed().as_secs_f64();
    let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "  [{tag}] Done: {} cols, {mb:.1} MB, {elapsed:.0}s",
        columns.len() + extra_cols.len()
    );
    Ok(Some(out_path))
}
