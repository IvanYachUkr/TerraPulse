//! Feature extraction module: reads seasonal GeoTIFFs, runs feature extraction,
//! writes output as parquet.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::features;
use crate::parquet_io;

const SEASONS: [&str; 3] = ["spring", "summer", "autumn"];
const NODATA: f32 = -9999.0;

/// Load a GeoTIFF as raw f32 band data using a Python helper.
/// Returns (n_bands, height, width, data) where data is [bands * H * W] flat.
fn read_tif_bands(tif_path: &Path) -> Result<(usize, usize, usize, Vec<f32>)> {
    // Use Python one-liner to read TIF and write raw binary + metadata
    let out = Command::new("python")
        .arg("-c")
        .arg(format!(
            r#"import rasterio, sys, struct
ds = rasterio.open(r'{}')
nb, h, w = ds.count, ds.height, ds.width
# Read only spectral bands (first 10), skip valid_fraction (band 11)
nb_read = min(nb, {})
sys.stdout.buffer.write(struct.pack('III', nb_read, h, w))
for b in range(1, nb_read + 1):
    arr = ds.read(b)
    sys.stdout.buffer.write(arr.astype('<f4').tobytes())
ds.close()"#,
            tif_path.display(),
            features::N_BANDS,
        ))
        .output()
        .context("Failed to run Python TIF reader")?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("Python TIF reader failed: {stderr}");
    }

    let buf = &out.stdout;
    if buf.len() < 12 {
        anyhow::bail!("TIF reader output too short ({} bytes)", buf.len());
    }

    let nb = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let h = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let w = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;

    let expected = 12 + nb * h * w * 4;
    if buf.len() < expected {
        anyhow::bail!(
            "TIF reader: expected {} bytes, got {} (nb={}, h={}, w={})",
            expected, buf.len(), nb, h, w,
        );
    }

    // Convert bytes to f32 (little-endian)
    let data: Vec<f32> = buf[12..expected]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok((nb, h, w, data))
}

/// Load valid fraction from band 11 of the TIF.
fn read_valid_fraction(tif_path: &Path) -> Result<Option<Vec<f32>>> {
    let out = Command::new("python")
        .arg("-c")
        .arg(format!(
            r#"import rasterio, sys, struct, numpy as np
ds = rasterio.open(r'{}')
if ds.count >= 11:
    vf = ds.read(11).astype('<f4')
    vf = np.where(vf > -9000, vf, np.nan)
    sys.stdout.buffer.write(struct.pack('II', ds.height, ds.width))
    sys.stdout.buffer.write(vf.tobytes())
ds.close()"#,
            tif_path.display(),
        ))
        .output()
        .context("Failed to read valid fraction")?;

    let buf = &out.stdout;
    if buf.len() < 8 {
        return Ok(None);
    }

    let h = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let w = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let data: Vec<f32> = buf[8..]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if data.len() == h * w {
        Ok(Some(data))
    } else {
        Ok(None)
    }
}

/// Detect if data is in DN (> 1.0 scale) and compute scale factor.
fn detect_scale(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for &v in data.iter().take(10000) {
        if v.is_finite() && v > 0.0 && v != NODATA {
            sum += v as f64;
            n += 1;
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
            let tif = raw_dir.join(format!("sentinel2_{region_name}_{actual_year}_{season}.tif"));
            if !tif.exists() {
                println!("  [{tag}] WARNING: Missing {} -- skip", tif.display());
                return Ok(None);
            }
            jobs.push((actual_year, season));
        }
    }

    let t0 = std::time::Instant::now();

    // Load all seasonal rasters
    let mut spectral_list: Vec<Vec<f32>> = Vec::new();
    let mut suffixes = Vec::new();
    let mut nr = 0usize;
    let mut nc = 0usize;
    let mut vf_first: Option<Vec<f32>> = None;

    for (actual_year, season) in &jobs {
        let model_year = year_map.iter().find(|(a, _)| a == actual_year).unwrap().1;
        let tif = raw_dir.join(format!("sentinel2_{region_name}_{actual_year}_{season}.tif"));

        let (nb, h, w, mut data) = read_tif_bands(&tif)?;
        assert!(nb >= features::N_BANDS, "TIF has {nb} bands, need {}", features::N_BANDS);

        if nr == 0 {
            nr = h / features::GP;
            nc = w / features::GP;
            vf_first = read_valid_fraction(&tif)?;
        }

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
        println!("    Loaded {actual_year}_{season} -> {model_year}_{season}");
    }

    let n_cells = nr * nc;

    // Run extraction
    let t1 = std::time::Instant::now();
    let flat = features::extract_all_seasons(&spectral_list, nr, nc);
    let dt = t1.elapsed().as_secs_f64();
    println!("    Rust extraction: {dt:.1}s for {} seasons", jobs.len());
    drop(spectral_list);

    let n_seasons = suffixes.len();

    // Build column names
    let base_names = features::feature_names();
    let mut columns = Vec::with_capacity(n_seasons * features::N_FEAT + 3);
    for suffix in &suffixes {
        for name in &base_names {
            columns.push(format!("{name}_{suffix}"));
        }
    }

    // Build row data [n_cells][n_features_total]
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

    // Impute NaN values with column medians
    let mut nan_count = 0u64;
    for col in 0..n_total_feats {
        // Collect finite values for this column
        let mut vals: Vec<f32> = rows.iter()
            .map(|row| row[col])
            .filter(|v| v.is_finite())
            .collect();

        let has_nan = vals.len() < n_cells;
        if !has_nan {
            continue;
        }

        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if vals.is_empty() {
            0.0
        } else {
            vals[vals.len() / 2]
        };

        for row in rows.iter_mut() {
            if !row[col].is_finite() {
                row[col] = median;
                nan_count += 1;
            }
        }
    }
    if nan_count > 0 {
        println!("    Imputed {nan_count} NaN values");
    }

    // Add cell_id, valid_fraction columns
    let mut extra_cols = vec!["cell_id".to_string()];
    let mut extra_data: Vec<Vec<f32>> = vec![(0..n_cells as u32).map(|i| i as f32).collect()];

    if let Some(ref vf) = vf_first {
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
        extra_data.push(vf_cells.clone());
        extra_cols.push("low_valid_fraction".to_string());
        extra_data.push(vf_cells.iter().map(|&v| if v < min_valid_frac { 1.0 } else { 0.0 }).collect());
    }

    // Write parquet
    parquet_io::write_feature_parquet(&out_path, &extra_cols, &extra_data, &columns, &rows)?;

    let elapsed = t0.elapsed().as_secs_f64();
    let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
    println!("  [{tag}] Done: {} cols, {mb:.1} MB, {elapsed:.0}s", columns.len() + extra_cols.len());
    Ok(Some(out_path))
}
