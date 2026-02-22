//! SAR (Sentinel-1 GRD) download, compositing, and GeoTIFF output.
//!
//! S1 GRD on Planetary Computer is stored as COGs with GCPs (no regular
//! geo-transform). We parse the GCPs, fit an affine transform via
//! least-squares, then resample to the target anchor grid.

use anyhow::{Context, Result};
use reqwest::Client;
use std::path::Path;

use crate::cog::{self, PixelBbox};
use crate::composite::AnchorRef;
use crate::reproject::GeoTransform;
use crate::stac;

/// Maximum raw DN value for scaling S1 amplitudes to [0, 1].
const MAX_DN: f32 = 2000.0;

/// SAR nodata value (matches Python: -9999).
const SAR_NODATA: f32 = -9999.0;

// ---- GCP handling ----

/// A ground control point from TIFF metadata.
#[derive(Debug, Clone)]
struct Gcp {
    pixel_x: f64, // column in pixel space
    pixel_y: f64, // row in pixel space
    geo_x: f64,   // longitude (EPSG:4326)
    geo_y: f64,   // latitude (EPSG:4326)
}

/// Parse GCPs from TIFF tag 33922 and organize into a grid for piecewise
/// bilinear interpolation. S1 GRD GCPs form a regular grid (e.g. 10×21).
///
/// For inverse mapping (lon,lat → pixel), we search the GCP geo grid to find
/// the enclosing cell, then bilinearly interpolate pixel coords.
struct GcpGrid {
    /// GCP grid dimensions
    n_rows: usize,
    n_cols: usize,
    /// Sorted unique row positions in pixel space
    row_positions: Vec<f64>,
    /// Sorted unique col positions in pixel space
    col_positions: Vec<f64>,
    /// Grid of longitudes [n_rows][n_cols]
    lon_grid: Vec<Vec<f64>>,
    /// Grid of latitudes [n_rows][n_cols]
    lat_grid: Vec<Vec<f64>>,
}

impl GcpGrid {
    /// Build the GCP grid from a list of GCPs.
    fn from_gcps(gcps: &[Gcp]) -> Self {
        // Extract unique sorted row and col positions
        let mut rows: Vec<f64> = gcps.iter().map(|g| g.pixel_y).collect();
        let mut cols: Vec<f64> = gcps.iter().map(|g| g.pixel_x).collect();
        rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rows.dedup_by(|a, b| (*a - *b).abs() < 1.0);
        cols.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cols.dedup_by(|a, b| (*a - *b).abs() < 1.0);

        let nr = rows.len();
        let nc = cols.len();

        // Build lon/lat grids
        let mut lon_grid = vec![vec![f64::NAN; nc]; nr];
        let mut lat_grid = vec![vec![f64::NAN; nc]; nr];

        for gcp in gcps {
            // Find the grid indices for this GCP
            let ri = rows.iter().position(|&r| (r - gcp.pixel_y).abs() < 1.0);
            let ci = cols.iter().position(|&c| (c - gcp.pixel_x).abs() < 1.0);
            if let (Some(ri), Some(ci)) = (ri, ci) {
                lon_grid[ri][ci] = gcp.geo_x;
                lat_grid[ri][ci] = gcp.geo_y;
            }
        }

        Self {
            n_rows: nr,
            n_cols: nc,
            row_positions: rows,
            col_positions: cols,
            lon_grid,
            lat_grid,
        }
    }

    /// Inverse mapping: (lon, lat) → fractional source pixel (px, py).
    ///
    /// Searches through GCP geo-grid cells to find the enclosing quadrilateral,
    /// then computes the inverse bilinear mapping to get pixel coordinates.
    fn wgs84_to_pixel(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
        // Search all GCP cells (quadrilaterals in geo-space)
        for ri in 0..self.n_rows - 1 {
            for ci in 0..self.n_cols - 1 {
                let lon00 = self.lon_grid[ri][ci];
                let lon10 = self.lon_grid[ri][ci + 1];
                let lon01 = self.lon_grid[ri + 1][ci];
                let lon11 = self.lon_grid[ri + 1][ci + 1];

                let lat00 = self.lat_grid[ri][ci];
                let lat10 = self.lat_grid[ri][ci + 1];
                let lat01 = self.lat_grid[ri + 1][ci];
                let lat11 = self.lat_grid[ri + 1][ci + 1];

                // Quick bounding box check
                let min_lon = lon00.min(lon10).min(lon01).min(lon11);
                let max_lon = lon00.max(lon10).max(lon01).max(lon11);
                let min_lat = lat00.min(lat10).min(lat01).min(lat11);
                let max_lat = lat00.max(lat10).max(lat01).max(lat11);

                if lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat {
                    continue;
                }

                // Try inverse bilinear
                if let Some((u, v)) = inverse_bilinear(
                    lon, lat, lon00, lat00, lon10, lat10, lon01, lat01, lon11, lat11,
                ) {
                    if u >= -0.01 && u <= 1.01 && v >= -0.01 && v <= 1.01 {
                        let px = self.col_positions[ci]
                            + u * (self.col_positions[ci + 1] - self.col_positions[ci]);
                        let py = self.row_positions[ri]
                            + v * (self.row_positions[ri + 1] - self.row_positions[ri]);
                        return Some((px, py));
                    }
                }
            }
        }
        None
    }
}

/// Inverse bilinear interpolation.
///
/// Given a point (x, y) inside a quadrilateral defined by four corners
/// (x00,y00), (x10,y10), (x01,y01), (x11,y11), find (u, v) in [0,1]×[0,1]
/// such that bilinear(u, v) = (x, y).
///
/// Uses Newton's method for the general case.
fn inverse_bilinear(
    x: f64,
    y: f64,
    x00: f64,
    y00: f64,
    x10: f64,
    y10: f64,
    x01: f64,
    y01: f64,
    x11: f64,
    y11: f64,
) -> Option<(f64, f64)> {
    // Bilinear: P(u,v) = (1-u)(1-v)*P00 + u(1-v)*P10 + (1-u)v*P01 + uv*P11
    // Rearrange: P = P00 + u(P10-P00) + v(P01-P00) + uv(P00-P10-P01+P11)
    let ax = x10 - x00;
    let ay = y10 - y00;
    let bx = x01 - x00;
    let by = y01 - y00;
    let cx = x00 - x10 - x01 + x11;
    let cy = y00 - y10 - y01 + y11;
    let dx = x - x00;
    let dy = y - y00;

    // Newton iteration to solve:
    //   ax*u + bx*v + cx*u*v = dx
    //   ay*u + by*v + cy*u*v = dy
    let mut u = 0.5;
    let mut v = 0.5;

    for _ in 0..20 {
        let fx = ax * u + bx * v + cx * u * v - dx;
        let fy = ay * u + by * v + cy * u * v - dy;

        if fx.abs() < 1e-10 && fy.abs() < 1e-10 {
            return Some((u, v));
        }

        // Jacobian
        let j11 = ax + cx * v;
        let j12 = bx + cx * u;
        let j21 = ay + cy * v;
        let j22 = by + cy * u;

        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-15 {
            return None;
        }

        u -= (j22 * fx - j12 * fy) / det;
        v -= (-j21 * fx + j11 * fy) / det;
    }

    // Check convergence
    let fx = ax * u + bx * v + cx * u * v - dx;
    let fy = ay * u + by * v + cy * u * v - dy;
    if fx.abs() < 1e-6 && fy.abs() < 1e-6 {
        Some((u, v))
    } else {
        None
    }
}

// ---- Parse GCPs from raw TIFF bytes ----

/// Read GCPs from a TIFF via HTTP. We fetch just the header + IFD to get
/// the ModelTiepoint tag (33922), which encodes GCPs as:
///   [I, J, K, X, Y, Z, I, J, K, X, Y, Z, ...]
/// where I,J = pixel coords and X,Y = geo coords.
async fn read_gcps_from_cog(client: &Client, url: &str) -> Result<Vec<Gcp>> {
    // Read the COG metadata (to validate the URL is accessible)
    let _meta = cog::read_cog_meta(client, url)
        .await
        .context("Failed to read S1 COG metadata for GCPs")?;

    // For GCP-referenced TIFFs, the tiepoint array has more than 6 elements.
    // Standard tiepoint: [I, J, K, X, Y, Z] (6 values)
    // Multiple GCPs: [I1, J1, K1, X1, Y1, Z1, I2, J2, K2, X2, Y2, Z2, ...]
    // Our COG reader currently only reads 6 values. We need to read the full
    // tiepoint array directly.

    // Read the full tiepoint from the raw IFD
    let header_bytes = cog::download_range(client, url, 0, 65536).await?;
    let gcps = parse_tiepoints_from_ifd(&header_bytes)?;

    if gcps.is_empty() {
        anyhow::bail!("No GCPs found in S1 scene");
    }

    Ok(gcps)
}

/// Parse tiepoints from raw TIFF IFD bytes.
fn parse_tiepoints_from_ifd(bytes: &[u8]) -> Result<Vec<Gcp>> {
    if bytes.len() < 8 {
        anyhow::bail!("TIFF header too short");
    }

    let le = bytes[0] == b'I' && bytes[1] == b'I';
    let read_u16 = if le {
        |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]])
    } else {
        |b: &[u8], o: usize| u16::from_be_bytes([b[o], b[o + 1]])
    };
    let read_u32 = if le {
        |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    } else {
        |b: &[u8], o: usize| u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    };
    let read_f64 = if le {
        |b: &[u8], o: usize| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&b[o..o + 8]);
            f64::from_le_bytes(arr)
        }
    } else {
        |b: &[u8], o: usize| {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&b[o..o + 8]);
            f64::from_be_bytes(arr)
        }
    };

    let ifd_offset = read_u32(bytes, 4) as usize;
    if ifd_offset + 2 > bytes.len() {
        anyhow::bail!("IFD offset out of range");
    }

    let n_entries = read_u16(bytes, ifd_offset) as usize;
    let mut gcps = Vec::new();

    for i in 0..n_entries {
        let entry_off = ifd_offset + 2 + i * 12;
        if entry_off + 12 > bytes.len() {
            break;
        }

        let tag = read_u16(bytes, entry_off);
        if tag != 33922 {
            // Not ModelTiepointTag
            continue;
        }

        let type_id = read_u16(bytes, entry_off + 2);
        let count = read_u32(bytes, entry_off + 4) as usize;

        // Type 12 = DOUBLE (8 bytes each)
        if type_id != 12 {
            continue;
        }

        let n_bytes = count * 8;
        let data_offset = if n_bytes <= 4 {
            entry_off + 8
        } else {
            read_u32(bytes, entry_off + 8) as usize
        };

        if data_offset + n_bytes > bytes.len() {
            // Data is beyond what we fetched — we need more bytes
            // For S1 with 210 GCPs: 210 * 6 * 8 = 10080 bytes
            // This should fit in our 64K header fetch
            break;
        }

        // Parse tiepoint triples: [I, J, K, X, Y, Z, ...]
        let n_gcps = count / 6;
        for g in 0..n_gcps {
            let off = data_offset + g * 6 * 8;
            let i_val = read_f64(bytes, off);
            let j_val = read_f64(bytes, off + 8);
            // K at off + 16 (skip)
            let x_val = read_f64(bytes, off + 24);
            let y_val = read_f64(bytes, off + 32);
            // Z at off + 40 (skip)

            gcps.push(Gcp {
                pixel_x: i_val,
                pixel_y: j_val,
                geo_x: x_val,
                geo_y: y_val,
            });
        }
        break;
    }

    Ok(gcps)
}

// ---- Main SAR download + composite ----

/// Download and composite SAR data for one season.
pub async fn download_sar_composite(
    client: &Client,
    items: &[stac::StacItem],
    token: &str,
    anchor: &AnchorRef,
    output_path: &Path,
) -> Result<()> {
    let dst_w = anchor.width;
    let dst_h = anchor.height;
    let n_pixels = dst_w * dst_h;
    let gt = &anchor.geo_transform;
    let (utm_zone, is_north) = utm_zone_info_from_epsg(anchor.epsg);

    let dst_gt = gt.clone();

    // Download scenes with limited concurrency (SAR scenes are large!)
    use futures::stream::{self, StreamExt};
    let max_concurrent = 4usize;

    let results: Vec<_> = stream::iter(items.iter().enumerate().map(|(si, item)| {
        let client = client.clone();
        let token = token.to_string();
        let dst_gt = dst_gt;
        let utm_z = utm_zone;
        let dw = dst_w;
        let dh = dst_h;
        let item = item.clone();

        async move {
            let max_retries = 2u32;
            let mut last_err = String::new();

            for attempt in 0..=max_retries {
                if attempt > 0 {
                    eprintln!(
                        "    SAR scene {}: retry {}/{}...",
                        si + 1,
                        attempt,
                        max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(3 * attempt as u64)).await;
                }
                match download_one_sar_scene(&client, &item, &token, dw, dh, &dst_gt, utm_z, is_north).await {
                    Ok(Some(data)) => {
                        eprintln!("    SAR scene {}: OK", si + 1);
                        return Ok(Some(data));
                    }
                    Ok(None) => {
                        eprintln!("    SAR scene {}: no overlap, skipped", si + 1);
                        return Ok(None);
                    }
                    Err(e) => {
                        last_err = format!("{e:#}");
                        if attempt == max_retries {
                            eprintln!("    SAR scene {}: FAILED - {}", si + 1, last_err);
                        }
                    }
                }
            }
            Err(anyhow::anyhow!("SAR scene {} failed: {}", si + 1, last_err))
        }
    }))
    .buffer_unordered(max_concurrent)
    .collect()
    .await;

    // Collect successful scenes
    let mut scenes: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let mut n_failed = 0;
    let mut n_skipped = 0;
    for r in results {
        match r {
            Ok(Some(data)) => scenes.push(data),
            Ok(None) => n_skipped += 1,
            Err(_) => n_failed += 1,
        }
    }

    eprintln!(
        "    {}/{} SAR scenes OK, {} skipped, {} failed",
        scenes.len(),
        scenes.len() + n_skipped + n_failed,
        n_skipped,
        n_failed
    );

    if scenes.is_empty() {
        anyhow::bail!("All SAR scenes failed");
    }

    // Compute nan-median composite for VV and VH
    let mut vv_composite = vec![f32::NAN; n_pixels];
    let mut vh_composite = vec![f32::NAN; n_pixels];

    for px in 0..n_pixels {
        // Collect finite values
        let mut vv_vals: Vec<f32> = scenes
            .iter()
            .map(|(vv, _)| vv[px])
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        let mut vh_vals: Vec<f32> = scenes
            .iter()
            .map(|(_, vh)| vh[px])
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();

        if !vv_vals.is_empty() {
            vv_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vv_composite[px] = median_sorted(&vv_vals);
        }
        if !vh_vals.is_empty() {
            vh_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vh_composite[px] = median_sorted(&vh_vals);
        }
    }

    // Scale: raw DN → clip [0, MAX_DN] → / MAX_DN → [0, 1]
    let mut vv_scaled = vec![SAR_NODATA; n_pixels];
    let mut vh_scaled = vec![SAR_NODATA; n_pixels];

    for px in 0..n_pixels {
        if vv_composite[px].is_finite() && vv_composite[px] > 0.0 {
            vv_scaled[px] = (vv_composite[px] / MAX_DN).clamp(0.0, 1.0);
        }
        if vh_composite[px].is_finite() && vh_composite[px] > 0.0 {
            vh_scaled[px] = (vh_composite[px] / MAX_DN).clamp(0.0, 1.0);
        }
    }

    // Write 2-band GeoTIFF
    write_sar_tif(output_path, &vv_scaled, &vh_scaled, dst_w, dst_h, anchor)?;

    Ok(())
}

fn median_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Download one S1 scene and resample VV/VH to target grid.
/// Returns Some((vv, vh)) if scene overlaps target area, None if it doesn't.
async fn download_one_sar_scene(
    client: &Client,
    item: &stac::StacItem,
    token: &str,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
    utm_zone: u32,
    is_north: bool,
) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
    let vv_asset = item.assets.get("vv").context("Missing VV asset")?;
    let vh_asset = item.assets.get("vh").context("Missing VH asset")?;

    let vv_url = stac::apply_token_pub(&vv_asset.href, token);
    let vh_url = stac::apply_token_pub(&vh_asset.href, token);

    // Read GCPs from VV band
    let gcps = read_gcps_from_cog(client, &vv_url).await?;
    let gcp_grid = GcpGrid::from_gcps(&gcps);

    // Read COG metadata for both bands
    let vv_meta = cog::read_cog_meta(client, &vv_url).await?;
    let vh_meta = cog::read_cog_meta(client, &vh_url).await?;

    // Check overlap: use GCP grid to verify the target area corners map to valid source pixels.
    // If none of the target area corners fall within the GCP grid, the scene doesn't cover the area.
    let corners = [
        (0.0, 0.0),
        (dst_w as f64, 0.0),
        (0.0, dst_h as f64),
        (dst_w as f64, dst_h as f64),
        (dst_w as f64 / 2.0, dst_h as f64 / 2.0), // center point too
    ];

    let mut has_overlap = false;
    for &(dx, dy) in &corners {
        let (gx, gy) = dst_gt.pixel_to_geo(dx, dy);
        let (lon, lat) = utm_to_wgs84(gx, gy, utm_zone, is_north);
        if gcp_grid.wgs84_to_pixel(lon, lat).is_some() {
            has_overlap = true;
            break;
        }
    }

    if !has_overlap {
        return Ok(None);
    }

    // Use GCP grid to find the source pixel bbox that covers our target area.
    // Sample a grid of target pixels and find the source pixel range.
    let step = 50; // sample every 50 pixels for speed
    let mut src_x0 = u32::MAX;
    let mut src_y0 = u32::MAX;
    let mut src_x1 = 0u32;
    let mut src_y1 = 0u32;

    for dy in (0..dst_h).step_by(step) {
        for dx in (0..dst_w).step_by(step) {
            let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);
            let (lon, lat) = utm_to_wgs84(gx, gy, utm_zone, is_north);
            if let Some((px, py)) = gcp_grid.wgs84_to_pixel(lon, lat) {
                src_x0 = src_x0.min(px as u32);
                src_y0 = src_y0.min(py as u32);
                src_x1 = src_x1.max(px as u32 + 1);
                src_y1 = src_y1.max(py as u32 + 1);
            }
        }
    }

    // Add padding and clamp
    let pad = 500u32;
    src_x0 = src_x0.saturating_sub(pad);
    src_y0 = src_y0.saturating_sub(pad);
    src_x1 = (src_x1 + pad).min(vv_meta.width);
    src_y1 = (src_y1 + pad).min(vv_meta.height);

    if src_x1 <= src_x0 || src_y1 <= src_y0 {
        return Ok(None);
    }

    let bbox = PixelBbox {
        x0: src_x0,
        y0: src_y0,
        x1: src_x1,
        y1: src_y1,
    };
    let crop_w = (src_x1 - src_x0) as usize;
    let crop_h = (src_y1 - src_y0) as usize;

    // Download VV and VH tiles concurrently
    let (vv_pixels, vh_pixels) = tokio::join!(
        cog::read_cog_region(client, &vv_url, &vv_meta, bbox),
        cog::read_cog_region(client, &vh_url, &vh_meta, bbox),
    );
    let vv_pixels = vv_pixels.context("VV download failed")?;
    let vh_pixels = vh_pixels.context("VH download failed")?;

    // Resample from GCP-referenced pixel space to target UTM grid
    let vv_resampled = resample_gcp_to_utm(
        &vv_pixels, crop_w, crop_h, src_x0, src_y0, &gcp_grid, dst_w, dst_h, dst_gt, utm_zone, is_north,
    );
    let vh_resampled = resample_gcp_to_utm(
        &vh_pixels, crop_w, crop_h, src_x0, src_y0, &gcp_grid, dst_w, dst_h, dst_gt, utm_zone, is_north,
    );

    Ok(Some((vv_resampled, vh_resampled)))
}

/// Resample a SAR band from GCP pixel space to UTM target grid.
///
/// For each target pixel:
///   1. UTM coord → WGS84 (lon, lat)
///   2. WGS84 → source pixel via inverse GCP transform
///   3. Bilinear sample from source raster
fn resample_gcp_to_utm(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    src_x_offset: u32,
    src_y_offset: u32,
    gcp_grid: &GcpGrid,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
    utm_zone: u32,
    is_north: bool,
) -> Vec<f32> {
    use rayon::prelude::*;

    let mut output = vec![f32::NAN; dst_w * dst_h];

    output
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            for dx in 0..dst_w {
                // 1. Target pixel center → UTM
                let (utm_x, utm_y) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);

                // 2. UTM → WGS84
                let (lon, lat) = utm_to_wgs84(utm_x, utm_y, utm_zone, is_north);

                // 3. WGS84 → source pixel via GCP grid (piecewise bilinear inverse)
                let (src_px, src_py) = match gcp_grid.wgs84_to_pixel(lon, lat) {
                    Some(p) => p,
                    None => continue,
                };

                // Apply crop offset
                let sx = src_px - src_x_offset as f64 - 0.5;
                let sy = src_py - src_y_offset as f64 - 0.5;

                // Bounds check
                if sx < -0.5 || sy < -0.5 || sx >= src_w as f64 - 0.5 || sy >= src_h as f64 - 0.5 {
                    continue;
                }

                // 4. Bilinear interpolation
                let x0 = sx.floor() as isize;
                let y0 = sy.floor() as isize;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let sample = |r: isize, c: isize| -> f64 {
                    if r < 0 || c < 0 || r >= src_h as isize || c >= src_w as isize {
                        return f64::NAN;
                    }
                    let v = src[r as usize * src_w + c as usize];
                    if v.is_finite() && v > 0.0 {
                        v as f64
                    } else {
                        f64::NAN
                    }
                };

                let v00 = sample(y0, x0);
                let v10 = sample(y0, x0 + 1);
                let v01 = sample(y0 + 1, x0);
                let v11 = sample(y0 + 1, x0 + 1);

                // Naive bilinear_interp fails completely if any of the 4 inputs is NaN.
                // We use a NaN-resilient method that normalizes the weights of the valid pixels,
                // matching rasterio/GDAL WarpedVRT behavior closely to prevent 42K missing edge pixels.
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;

                let mut sum = 0.0;
                let mut wsum = 0.0;

                if v00.is_finite() {
                    sum += v00 * w00;
                    wsum += w00;
                }
                if v10.is_finite() {
                    sum += v10 * w10;
                    wsum += w10;
                }
                if v01.is_finite() {
                    sum += v01 * w01;
                    wsum += w01;
                }
                if v11.is_finite() {
                    sum += v11 * w11;
                    wsum += w11;
                }

                row[dx] = if wsum > 1e-9 {
                    (sum / wsum) as f32
                } else {
                    f32::NAN
                };
            }
        });

    output
}

/// UTM → WGS84 inverse projection (approximate, good to ~1m accuracy).
fn utm_to_wgs84(easting: f64, northing: f64, zone: u32, is_north: bool) -> (f64, f64) {
    use std::f64::consts::PI;

    let a: f64 = 6378137.0;
    let f: f64 = 1.0 / 298.257223563;
    let e2: f64 = 2.0 * f - f * f;
    let e1: f64 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());
    let k0: f64 = 0.9996;
    let e_prime2: f64 = e2 / (1.0 - e2);

    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0) * PI / 180.0;

    let x = easting - 500000.0; // remove false easting
    let y = if is_north { northing } else { northing - 10_000_000.0 }; // subtract false northing for southern hemisphere

    let m = y / k0;
    let mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2.powi(2) / 64.0 - 5.0 * e2.powi(3) / 256.0));

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin();

    let n1 = a / (1.0 - e2 * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan().powi(2);
    let c1 = e_prime2 * phi1.cos().powi(2);
    let r1 = a * (1.0 - e2) / (1.0 - e2 * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * k0);

    let lat = phi1
        - (n1 * phi1.tan() / r1)
            * (d.powi(2) / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * e_prime2) * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * e_prime2
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);

    let lon = lon0
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2) + 8.0 * e_prime2 + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / phi1.cos();

    (lon * 180.0 / PI, lat * 180.0 / PI)
}

/// Write a 2-band SAR GeoTIFF.
/// Determine UTM zone and hemisphere from EPSG code.
fn utm_zone_info_from_epsg(epsg: u32) -> (u32, bool) {
    if epsg >= 32601 && epsg <= 32660 {
        (epsg - 32600, true) // Northern hemisphere
    } else if epsg >= 32701 && epsg <= 32760 {
        (epsg - 32700, false) // Southern hemisphere
    } else {
        (32, true) // fallback for central Europe (north)
    }
}

fn write_sar_tif(
    path: &Path,
    vv: &[f32],
    vh: &[f32],
    width: usize,
    height: usize,
    anchor: &AnchorRef,
) -> Result<()> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    let n_pixels = width * height;
    let n_bands = 2u16;

    // Build pixel data first: interleaved [VV_0, VH_0, VV_1, VH_1, ...]
    let mut pixel_bytes = Vec::with_capacity(n_pixels * 8); // 2 bands * 4 bytes
    for i in 0..n_pixels {
        pixel_bytes.extend_from_slice(&vv[i].to_le_bytes());
        pixel_bytes.extend_from_slice(&vh[i].to_le_bytes());
    }

    // Compress with zlib-wrapped DEFLATE (TIFF Compression=8)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&pixel_bytes)?;
    let compressed = encoder.finish()?;
    let compressed_bytes = compressed.len() as u32;

    // Build GeoTIFF with same structure as composite.rs
    let gt = &anchor.geo_transform;
    let epsg = anchor.epsg;

    let mut buf = Vec::new();

    // TIFF header
    buf.write_all(b"II")?; // little-endian
    buf.write_all(&42u16.to_le_bytes())?;
    let ifd_offset = 8u32;
    buf.write_all(&ifd_offset.to_le_bytes())?;

    // IFD entries
    let n_entries = 15u16;
    buf.write_all(&n_entries.to_le_bytes())?;

    let data_offset = 8 + 2 + n_entries as u32 * 12 + 4;

    // Extra data after IFD
    let pixel_scale_data: [f64; 3] = [gt.pixel_size_x, gt.pixel_size_y, 0.0];
    let tiepoint_data: [f64; 6] = [0.0, 0.0, 0.0, gt.origin_x, gt.origin_y, 0.0];

    // GeoKeys for UTM
    let geo_key_data: [u16; 16] = [
        1,
        1,
        0,
        3, // KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys
        1024,
        0,
        1,
        1, // GTModelTypeGeoKey = ModelTypeProjected
        1025,
        0,
        1,
        1, // GTRasterTypeGeoKey = RasterPixelIsArea
        3072,
        0,
        1,
        epsg as u16, // ProjectedCSTypeGeoKey
    ];

    let nodata_str = b"-9999\0";

    // Compute offsets for extra data
    let pixel_scale_off = data_offset;
    let tiepoint_off = pixel_scale_off + 24; // 3 * f64
    let geo_key_off = tiepoint_off + 48; // 6 * f64
    let nodata_off = geo_key_off + 32; // 16 * u16
    let strip_data_off = nodata_off + nodata_str.len() as u32;

    // BitsPerSample inline: two u16 values (32, 32) packed into a u32 (little-endian)
    let bps_inline: u32 = 32u32 | (32u32 << 16);
    // SampleFormat inline: two u16 values (3, 3) packed into a u32 (IEEEFP=3)
    let sf_inline: u32 = 3u32 | (3u32 << 16);

    // Write IFD entries (must be sorted by tag number!)
    let ifd_entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, width as u32),                       // ImageWidth
        (257, 4, 1, height as u32),                      // ImageLength
        (258, 3, 2, bps_inline),                         // BitsPerSample = [32, 32] inline
        (259, 3, 1, 8),                                  // Compression = DEFLATE
        (262, 3, 1, 1),                                  // PhotometricInterpretation = MinIsBlack
        (273, 4, 1, strip_data_off),                     // StripOffsets (1 strip)
        (277, 3, 1, n_bands as u32),                     // SamplesPerPixel = 2
        (278, 4, 1, height as u32),                      // RowsPerStrip
        (279, 4, 1, compressed_bytes),                   // StripByteCounts = compressed
        (284, 3, 1, 1),                  // PlanarConfiguration = Chunky (interleaved)
        (339, 3, 2, sf_inline),          // SampleFormat = [IEEEFP, IEEEFP] inline
        (33550, 12, 3, pixel_scale_off), // ModelPixelScaleTag
        (33922, 12, 6, tiepoint_off),    // ModelTiepointTag
        (34735, 3, 16, geo_key_off),     // GeoKeyDirectoryTag
        (42113, 2, nodata_str.len() as u32, nodata_off), // GDAL_NODATA
    ];

    for (tag, type_id, count, value) in &ifd_entries {
        buf.write_all(&tag.to_le_bytes())?;
        buf.write_all(&type_id.to_le_bytes())?;
        buf.write_all(&count.to_le_bytes())?;
        buf.write_all(&value.to_le_bytes())?;
    }

    // Next IFD = 0
    buf.write_all(&0u32.to_le_bytes())?;

    // Write extended data
    for &v in &pixel_scale_data {
        buf.write_all(&v.to_le_bytes())?;
    }
    for &v in &tiepoint_data {
        buf.write_all(&v.to_le_bytes())?;
    }
    for &v in &geo_key_data {
        buf.write_all(&v.to_le_bytes())?;
    }
    buf.write_all(nodata_str)?;

    // Write compressed pixel data
    buf.extend_from_slice(&compressed);

    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(path, &buf)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_sorted() {
        let empty: &[f32] = &[];
        assert!(median_sorted(empty).is_nan());

        let one = &[5.0];
        assert_eq!(median_sorted(one), 5.0);

        let two = &[2.0, 4.0];
        assert_eq!(median_sorted(two), 3.0);

        let three = &[1.0, 10.0, 100.0];
        assert_eq!(median_sorted(three), 10.0);

        let four = &[1.0, 2.0, 3.0, 4.0];
        assert_eq!(median_sorted(four), 2.5);
    }

    #[test]
    fn test_utm_to_wgs84() {
        // Test coordinate: Center of Nuremberg
        // lat: 49.4521, lon: 11.0767
        // UTM Zone 32N
        // Expected Easting: ~650630, Expected Northing: ~5479630
        
        // Exact derived coordinates from EPSG:32632
        let easting = 650630.0;
        let northing = 5479630.0;
        let zone = 32;

        let (lon, lat) = utm_to_wgs84(easting, northing, zone, true);

        // Approximate inversion for 32N coordinates (49.4506, 11.0782)
        assert!(
            (lat - 49.450644).abs() < 1e-5 && (lon - 11.078287).abs() < 1e-5,
            "Expected (~49.450644, ~11.078287), got ({}, {})",
            lat, lon
        );
    }

    #[test]
    fn test_utm_to_wgs84_southern() {
        // Cape Town area: lat ~-33.9, lon ~18.4
        // UTM Zone 34S, EPSG:32734
        let easting = 261878.0;
        let northing = 6243186.0;
        let zone = 34;
        let (lon, lat) = utm_to_wgs84(easting, northing, zone, false);
        assert!(
            (lat - (-33.93)).abs() < 0.05 && (lon - 18.42).abs() < 0.05,
            "Southern hemisphere: expected (~-33.93, ~18.42), got ({}, {})",
            lat, lon
        );
        // Verify latitude is negative (southern hemisphere)
        assert!(lat < 0.0, "Southern hemisphere latitude should be negative, got {}", lat);
    }

    #[test]
    fn test_utm_zone_info_from_epsg() {
        assert_eq!(utm_zone_info_from_epsg(32632), (32, true));  // 32N
        assert_eq!(utm_zone_info_from_epsg(32633), (33, true));  // 33N
        assert_eq!(utm_zone_info_from_epsg(32732), (32, false)); // 32S
        assert_eq!(utm_zone_info_from_epsg(4326), (32, true));   // fallback
    }
}
