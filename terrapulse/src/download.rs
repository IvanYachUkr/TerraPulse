use anyhow::Result;
use reqwest::Client;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::composite::{self, AnchorRef};
use crate::stac;

const MIN_SCENES: usize = 8;

/// Download one season's composite and write it to a GeoTIFF.
///
/// Pure Rust path: STAC search → sign URLs → download COG tiles → reproject →
/// cloud mask → nanmedian composite → write GeoTIFF.
pub async fn download_season(
    client: &Client,
    bbox: [f64; 4],
    _epsg: u32,
    year: u32,
    season: &str,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<Option<PathBuf>> {
    let out_path = raw_dir.join(format!("sentinel2_{region_name}_{year}_{season}.tif"));
    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [{year}/{season}] Already exists ({mb:.1} MB) -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(raw_dir)?;

    // 1. Search for scenes
    eprintln!("  [{year}/{season}] Searching STAC...");
    let mut items = stac::search_with_fallback(client, bbox, year, season, MIN_SCENES).await?;
    if items.is_empty() {
        eprintln!("  [{year}/{season}] WARNING: No scenes found -- skipping!");
        return Ok(None);
    }

    // Cap scenes to prevent OOM on orbit-dense regions (e.g. Crete, equatorial).
    // STAC results are sorted by cloud cover, so truncating keeps the best scenes.
    const MAX_SCENES: usize = 20;
    if items.len() > MAX_SCENES {
        eprintln!(
            "  [{year}/{season}] Capping {} scenes to {MAX_SCENES} (lowest cloud cover)",
            items.len()
        );
        items.truncate(MAX_SCENES);
    }

    eprintln!(
        "  [{year}/{season}] Using {} scenes, signing...",
        items.len()
    );

    // 2. Get collection SAS token and sign all URLs
    let all_bands = stac::all_download_bands();
    let token = stac::get_collection_token(client).await?;
    let signed_scenes: Vec<HashMap<String, String>> = items
        .iter()
        .map(|item| {
            let band_refs: Vec<&str> = all_bands.iter().copied().collect();
            stac::sign_scene_assets_with_token(item, &band_refs, &token)
        })
        .collect::<Result<Vec<_>>>()?;

    // 3. Download, reproject, composite in pure Rust
    eprintln!("  [{year}/{season}] Downloading and compositing (pure Rust)...");
    composite::download_and_composite(client, &items, &signed_scenes, anchor, &out_path, year)
        .await?;

    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [{year}/{season}] Written ({mb:.1} MB)");
        Ok(Some(out_path))
    } else {
        anyhow::bail!("Composite failed to produce {}", out_path.display());
    }
}

/// Download all seasons for a year — concurrently (3 seasons at once).
pub async fn download_year(
    client: &Client,
    bbox: [f64; 4],
    epsg: u32,
    year: u32,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<()> {
    let (r1, r2, r3) = tokio::join!(
        download_season(client, bbox, epsg, year, "spring", region_name, raw_dir, anchor),
        download_season(client, bbox, epsg, year, "summer", region_name, raw_dir, anchor),
        download_season(client, bbox, epsg, year, "autumn", region_name, raw_dir, anchor),
    );
    r1?;
    r2?;
    r3?;
    Ok(())
}


// ---- SAR (Sentinel-1) download ----

/// Download one season of SAR composite.
pub async fn download_sar_season(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    season: &str,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<Option<PathBuf>> {
    let out_path = raw_dir.join(format!("sentinel1_{region_name}_{year}_{season}.tif"));
    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [SAR {year}/{season}] Already exists ({mb:.1} MB) -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(raw_dir)?;

    // 1. Search for S1 scenes
    eprintln!("  [SAR {year}/{season}] Searching STAC...");
    let items = stac::search_sar_scenes(client, bbox, year, season).await?;
    if items.is_empty() {
        eprintln!("  [SAR {year}/{season}] WARNING: No S1 scenes found -- skipping!");
        return Ok(None);
    }
    eprintln!(
        "  [SAR {year}/{season}] Found {} scenes, downloading...",
        items.len()
    );

    // 2. Get S1 SAS token
    let token = stac::get_s1_token(client).await?;

    // 3. Download, resample, composite in pure Rust
    crate::sar_download::download_sar_composite(client, &items, &token, anchor, &out_path).await?;

    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [SAR {year}/{season}] Written ({mb:.1} MB)");
        Ok(Some(out_path))
    } else {
        anyhow::bail!("SAR composite failed to produce {}", out_path.display());
    }
}

/// Download SAR for all seasons of a year — concurrently.
pub async fn download_sar_year(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<()> {
    let (r1, r2, r3) = tokio::join!(
        download_sar_season(client, bbox, year, "spring", region_name, raw_dir, anchor),
        download_sar_season(client, bbox, year, "summer", region_name, raw_dir, anchor),
        download_sar_season(client, bbox, year, "autumn", region_name, raw_dir, anchor),
    );
    r1?;
    r2?;
    r3?;
    Ok(())
}
