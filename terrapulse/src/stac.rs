//! STAC API client for Planetary Computer Sentinel-2 search + URL signing.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const STAC_API: &str = "https://planetarycomputer.microsoft.com/api/stac/v1";
const SIGN_API: &str = "https://planetarycomputer.microsoft.com/api/sas/v1/sign";

// ---- STAC search request/response types ----

#[derive(Serialize)]
struct StacSearchBody {
    collections: Vec<String>,
    bbox: [f64; 4],
    datetime: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<serde_json::Value>,
    limit: u32,
}

#[derive(Deserialize, Debug)]
pub struct StacFeatureCollection {
    pub features: Vec<StacItem>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StacItem {
    pub id: String,
    pub properties: StacProperties,
    pub assets: HashMap<String, StacAsset>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StacProperties {
    #[serde(rename = "eo:cloud_cover")]
    pub cloud_cover: Option<f64>,
    pub datetime: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct StacAsset {
    pub href: String,
    #[serde(rename = "type")]
    pub media_type: Option<String>,
}

#[derive(Deserialize)]
struct SignResponse {
    href: String,
}

// ---- Season date ranges ----

pub fn season_date_range(year: u32, season: &str) -> (String, String) {
    match season {
        "spring" => (format!("{year}-04-01"), format!("{year}-05-31")),
        "summer" => (format!("{year}-06-01"), format!("{year}-08-31")),
        "autumn" => (format!("{year}-09-01"), format!("{year}-10-31")),
        _ => panic!("Unknown season: {season}"),
    }
}

// ---- STAC search ----

/// Search for Sentinel-2 L2A scenes matching the given parameters.
pub async fn search_scenes(
    client: &Client,
    bbox: [f64; 4],
    start_date: &str,
    end_date: &str,
    cloud_max: f64,
) -> Result<Vec<StacItem>> {
    let body = StacSearchBody {
        collections: vec!["sentinel-2-l2a".to_string()],
        bbox,
        datetime: format!("{start_date}/{end_date}"),
        query: Some(serde_json::json!({
            "eo:cloud_cover": {"lt": cloud_max}
        })),
        limit: 500,
    };

    let url = format!("{STAC_API}/search");
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("STAC search request failed")?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("STAC search returned {status}: {text}");
    }

    let fc: StacFeatureCollection = resp.json().await.context("Failed to parse STAC response")?;
    Ok(fc.features)
}

/// Search with cloud cover ramp (40 -> 50 -> 60) and date expansion fallback.
pub async fn search_with_fallback(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    season: &str,
    min_scenes: usize,
) -> Result<Vec<StacItem>> {
    let (start, end) = season_date_range(year, season);

    // Try increasing cloud cover thresholds
    for cloud_max in [40.0, 50.0, 60.0] {
        let items = search_scenes(client, bbox, &start, &end, cloud_max).await?;
        if items.len() >= min_scenes {
            return Ok(items);
        }
    }

    // Expand date window by ±14 days
    let s = chrono_parse_expand(&start, -14);
    let e = chrono_parse_expand(&end, 14);
    let items = search_scenes(client, bbox, &s, &e, 60.0).await?;
    Ok(items)
}

/// Simple date expansion (±days) without pulling in chrono.
fn chrono_parse_expand(date_str: &str, days: i32) -> String {
    // Parse YYYY-MM-DD, add days naively
    let parts: Vec<u32> = date_str.split('-').map(|p| p.parse().unwrap()).collect();
    let (y, m, d) = (parts[0] as i32, parts[1] as i32, parts[2] as i32);

    // Convert to a rough day count and back (good enough for ±14 days)
    let days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut total_day = d + days;
    let mut month = m;
    let mut year = y;

    while total_day < 1 {
        month -= 1;
        if month < 1 {
            month = 12;
            year -= 1;
        }
        total_day += days_in_month[month as usize];
    }
    while total_day > days_in_month[month as usize] {
        total_day -= days_in_month[month as usize];
        month += 1;
        if month > 12 {
            month = 1;
            year += 1;
        }
    }

    format!("{year:04}-{month:02}-{total_day:02}")
}

// ---- URL signing ----

/// Sign a single Planetary Computer blob URL.
pub async fn sign_url(client: &Client, href: &str) -> Result<String> {
    let resp = client
        .get(SIGN_API)
        .query(&[("href", href)])
        .send()
        .await
        .context("Sign URL request failed")?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Sign URL returned {status}: {text}");
    }

    let sr: SignResponse = resp.json().await.context("Failed to parse sign response")?;
    Ok(sr.href)
}

/// Sign all band asset URLs in a scene item.
pub async fn sign_scene_assets(
    client: &Client,
    item: &StacItem,
    band_names: &[&str],
) -> Result<HashMap<String, String>> {
    let mut signed = HashMap::new();
    for band in band_names {
        let band_str = band.to_string();
        if let Some(asset) = item.assets.get(&band_str) {
            let signed_href = sign_url(client, &asset.href).await?;
            signed.insert(band_str, signed_href);
        } else {
            anyhow::bail!("Scene {} missing band {}", item.id, band);
        }
    }
    Ok(signed)
}

/// Get the list of band names we need to download.
pub fn sentinel_bands() -> Vec<&'static str> {
    vec!["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
}

/// Get bands + SCL for cloud masking.
pub fn all_download_bands() -> Vec<&'static str> {
    vec!["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
}
