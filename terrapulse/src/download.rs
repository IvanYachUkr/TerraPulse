//! Parallel Sentinel-2 band downloader + median compositing.
//!
//! Fetches COG data, does cloud masking, and writes composites via
//! a small Python/rasterio helper for reprojection + GeoTIFF writing.

use anyhow::{Context, Result};
use reqwest::Client;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::stac::{self, StacItem};

const SENTINEL_NODATA: f32 = -9999.0;
const SCL_EXCLUDE: [u8; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
const MIN_SCENES: usize = 8;

/// Download a single band from a signed URL. Returns raw bytes.
async fn download_band_bytes(client: &Client, signed_url: &str) -> Result<Vec<u8>> {
    let resp = client
        .get(signed_url)
        .send()
        .await
        .context("Band download failed")?;

    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("Band download returned {status}");
    }

    let bytes = resp.bytes().await.context("Failed to read band bytes")?;
    Ok(bytes.to_vec())
}

/// Download one season's composite and write it to a GeoTIFF.
///
/// This function:
/// 1. Searches STAC for scenes
/// 2. Signs all band URLs
/// 3. Downloads bands in parallel
/// 4. Delegates compositing + reprojection to a Python helper
pub async fn download_season(
    client: &Client,
    bbox: [f64; 4],
    _epsg: u32,
    year: u32,
    season: &str,
    region_name: &str,
    raw_dir: &Path,
    anchor_ref_path: &Path,
    helper_script: &Path,
) -> Result<Option<PathBuf>> {
    let out_path = raw_dir.join(format!("sentinel2_{region_name}_{year}_{season}.tif"));
    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        println!("  [{year}/{season}] Already exists ({mb:.1} MB) -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(raw_dir)?;

    // 1. Search for scenes
    println!("  [{year}/{season}] Searching STAC...");
    let items = stac::search_with_fallback(client, bbox, year, season, MIN_SCENES).await?;
    if items.is_empty() {
        println!("  [{year}/{season}] WARNING: No scenes found -- skipping!");
        return Ok(None);
    }
    println!("  [{year}/{season}] Found {} scenes, signing...", items.len());

    // 2. Get collection SAS token (single API call) and sign all URLs locally
    let all_bands = stac::all_download_bands();
    let token = stac::get_collection_token(client).await?;
    let signed_scenes: Vec<std::collections::HashMap<String, String>> = items
        .iter()
        .map(|item| {
            let band_refs: Vec<&str> = all_bands.iter().copied().collect();
            stac::sign_scene_assets_with_token(item, &band_refs, &token)
        })
        .collect::<Result<Vec<_>>>()?;
    println!("  [{year}/{season}] URLs signed, compositing...");

    // 3. Pass signed URLs to Python helper for download + composite + reproject
    // (Python uses rasterio WarpedVRT for efficient reprojection)
    let scene_json = build_scene_json(&items, &signed_scenes)?;
    run_composite_helper(helper_script, &scene_json, anchor_ref_path, &out_path, year)?;

    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        println!("  [{year}/{season}] Written ({mb:.1} MB)");
        Ok(Some(out_path))
    } else {
        anyhow::bail!("Composite helper failed to produce {}", out_path.display());
    }
}


/// Build JSON describing scenes + signed URLs for the Python helper.
fn build_scene_json(
    items: &[StacItem],
    signed_scenes: &[HashMap<String, String>],
) -> Result<String> {
    let scenes: Vec<serde_json::Value> = items
        .iter()
        .zip(signed_scenes.iter())
        .map(|(item, signed)| {
            serde_json::json!({
                "id": item.id,
                "bands": signed,
            })
        })
        .collect();

    serde_json::to_string(&scenes).context("Failed to serialize scene JSON")
}

/// Call the Python composite helper script.
fn run_composite_helper(
    helper_script: &Path,
    scene_json: &str,
    anchor_ref_path: &Path,
    output_path: &Path,
    year: u32,
) -> Result<()> {
    // Write scene JSON to a temp file to avoid command line length limits
    let tmp_json = output_path.with_extension("scenes.json");
    std::fs::write(&tmp_json, scene_json)?;

    let status = Command::new("python")
        .arg(helper_script)
        .arg("--scenes-json")
        .arg(&tmp_json)
        .arg("--anchor-ref")
        .arg(anchor_ref_path)
        .arg("--output")
        .arg(output_path)
        .arg("--year")
        .arg(year.to_string())
        .status()
        .context("Failed to run composite helper (is Python with rasterio installed?)")?;

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp_json);

    if !status.success() {
        anyhow::bail!("Composite helper exited with {status}");
    }
    Ok(())
}

/// Download all seasons for a year.
pub async fn download_year(
    client: &Client,
    bbox: [f64; 4],
    epsg: u32,
    year: u32,
    region_name: &str,
    raw_dir: &Path,
    anchor_ref_path: &Path,
    helper_script: &Path,
) -> Result<()> {
    for season in ["spring", "summer", "autumn"] {
        download_season(
            client,
            bbox,
            epsg,
            year,
            season,
            region_name,
            raw_dir,
            anchor_ref_path,
            helper_script,
        )
        .await?;
    }
    Ok(())
}
