//! STAC API client for Planetary Computer Sentinel-2 search + URL signing.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const STAC_API: &str = "https://planetarycomputer.microsoft.com/api/stac/v1";
const TOKEN_API: &str = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a";

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
#[allow(dead_code)]
pub struct StacItem {
    pub id: String,
    pub properties: StacProperties,
    pub assets: HashMap<String, StacAsset>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct StacProperties {
    #[serde(rename = "eo:cloud_cover")]
    pub cloud_cover: Option<f64>,
    pub datetime: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct StacAsset {
    pub href: String,
    #[serde(rename = "type")]
    pub media_type: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    token: String,
}

// ---- Season date ranges ----

pub fn season_date_range(year: u32, season: &str) -> Result<(String, String)> {
    match season {
        "spring" => Ok((format!("{year}-04-01"), format!("{year}-05-31"))),
        "summer" => Ok((format!("{year}-06-01"), format!("{year}-08-31"))),
        "autumn" => Ok((format!("{year}-09-01"), format!("{year}-10-31"))),
        _ => anyhow::bail!("Unknown season: '{season}' (expected spring/summer/autumn)"),
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
    let (start, end) = season_date_range(year, season)?;

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
    let days_in_month = |y: i32, m: i32| -> i32 {
        match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                    29
                } else {
                    28
                }
            }
            _ => 30,
        }
    };
    let mut total_day = d + days;
    let mut month = m;
    let mut year = y;

    while total_day < 1 {
        month -= 1;
        if month < 1 {
            month = 12;
            year -= 1;
        }
        total_day += days_in_month(year, month);
    }
    while total_day > days_in_month(year, month) {
        total_day -= days_in_month(year, month);
        month += 1;
        if month > 12 {
            month = 1;
            year += 1;
        }
    }

    format!("{year:04}-{month:02}-{total_day:02}")
}

// ---- Token-based signing ----

/// Cached SAS token with timestamp for auto-refresh.
static CACHED_TOKEN: tokio::sync::Mutex<Option<(String, std::time::Instant)>> =
    tokio::sync::Mutex::const_new(None);

/// Token TTL: refresh after 50 minutes (Planetary Computer tokens last ~1 hour).
const TOKEN_TTL_SECS: u64 = 50 * 60;

/// Fetch a fresh SAS token (with retry on 429).
async fn fetch_token(client: &Client, api_url: &str) -> Result<String> {
    let max_retries = 3;
    let mut wait_secs = 5u64;

    for attempt in 0..=max_retries {
        let resp = client
            .get(api_url)
            .send()
            .await
            .with_context(|| format!("Token request to {api_url} failed"))?;

        let status = resp.status();
        if status.as_u16() == 429 && attempt < max_retries {
            eprintln!("    Rate limited on token, waiting {wait_secs}s...");
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            wait_secs *= 2;
            continue;
        }

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Token API {api_url} returned {status}: {text}");
        }

        let tr: TokenResponse = resp
            .json()
            .await
            .context("Failed to parse token response")?;
        return Ok(tr.token);
    }

    anyhow::bail!("Token fetch exhausted retries for {api_url}")
}

/// Get a SAS token for the sentinel-2-l2a collection (cached, auto-refreshes after 50 min).
pub async fn get_collection_token(client: &Client) -> Result<String> {
    let mut guard = CACHED_TOKEN.lock().await;
    if let Some((ref token, ref ts)) = *guard {
        if ts.elapsed().as_secs() < TOKEN_TTL_SECS {
            return Ok(token.clone());
        }
        eprintln!("    S2 token expired, refreshing...");
    }
    let token = fetch_token(client, TOKEN_API).await?;
    *guard = Some((token.clone(), std::time::Instant::now()));
    Ok(token)
}

/// Apply a SAS token to a blob URL.
fn apply_token(href: &str, token: &str) -> String {
    if href.contains('?') {
        format!("{href}&{token}")
    } else {
        format!("{href}?{token}")
    }
}

/// Public version of apply_token for cross-module use.
pub fn apply_token_pub(href: &str, token: &str) -> String {
    apply_token(href, token)
}

/// Sign all band asset URLs in a scene item using a pre-fetched token.
pub fn sign_scene_assets_with_token(
    item: &StacItem,
    band_names: &[&str],
    token: &str,
) -> Result<HashMap<String, String>> {
    let mut signed = HashMap::new();
    for band in band_names {
        let band_str = band.to_string();
        if let Some(asset) = item.assets.get(&band_str) {
            signed.insert(band_str, apply_token(&asset.href, token));
        } else {
            anyhow::bail!("Scene {} missing band {}", item.id, band);
        }
    }
    Ok(signed)
}

/// Get bands + SCL for cloud masking.
pub fn all_download_bands() -> Vec<&'static str> {
    vec![
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL",
    ]
}

// ---- Sentinel-1 SAR ----

const S1_TOKEN_API: &str =
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd";

/// Cached SAS token for S1 with auto-refresh.
static CACHED_S1_TOKEN: tokio::sync::Mutex<Option<(String, std::time::Instant)>> =
    tokio::sync::Mutex::const_new(None);

/// Get a SAS token for sentinel-1-grd (cached, auto-refreshes after 50 min).
pub async fn get_s1_token(client: &Client) -> Result<String> {
    let mut guard = CACHED_S1_TOKEN.lock().await;
    if let Some((ref token, ref ts)) = *guard {
        if ts.elapsed().as_secs() < TOKEN_TTL_SECS {
            return Ok(token.clone());
        }
        eprintln!("    S1 token expired, refreshing...");
    }
    let token = fetch_token(client, S1_TOKEN_API).await?;
    *guard = Some((token.clone(), std::time::Instant::now()));
    Ok(token)
}

/// Search for Sentinel-1 IW GRD scenes.
pub async fn search_sar_scenes(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    season: &str,
) -> Result<Vec<StacItem>> {
    let (start, end) = season_date_range(year, season)?;

    // First try ascending orbit only
    let body = StacSearchBody {
        collections: vec!["sentinel-1-grd".to_string()],
        bbox,
        datetime: format!("{start}/{end}"),
        query: Some(serde_json::json!({
            "sar:instrument_mode": {"eq": "IW"},
            "sat:orbit_state": {"eq": "ascending"}
        })),
        limit: 500,
    };

    let url = format!("{STAC_API}/search");
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("S1 STAC search failed")?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("S1 STAC search returned {status}: {text}");
    }
    let fc: StacFeatureCollection = resp
        .json()
        .await
        .context("Failed to parse S1 STAC response")?;

    if fc.features.len() >= 3 {
        return Ok(fc.features);
    }

    // Fallback: any orbit
    let body2 = StacSearchBody {
        collections: vec!["sentinel-1-grd".to_string()],
        bbox,
        datetime: format!("{start}/{end}"),
        query: Some(serde_json::json!({
            "sar:instrument_mode": {"eq": "IW"}
        })),
        limit: 500,
    };

    let resp2 = client
        .post(&url)
        .json(&body2)
        .send()
        .await
        .context("S1 STAC fallback search failed")?;
    if !resp2.status().is_success() {
        return Ok(fc.features); // return whatever we had
    }
    let fc2: StacFeatureCollection = resp2
        .json()
        .await
        .context("Failed to parse S1 STAC fallback response")?;

    if fc2.features.len() > fc.features.len() {
        Ok(fc2.features)
    } else {
        Ok(fc.features)
    }
}
