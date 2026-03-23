use anyhow::Result;
use reqwest::Client;
use std::path::Path;
use serde_json::json;

use crate::composite::AnchorRef;
use crate::cog::{read_cog_meta, read_cog_region, PixelBbox};

const CLASS_NAMES: [&str; 7] = ["tree_cover", "shrubland", "grassland", "cropland", "built_up", "bare_sparse", "water"];
const N_CLASSES: usize = 7;

fn map_wc_class(code: u8) -> Option<usize> {
    match code {
        10 => Some(0),
        20 => Some(1),
        30 => Some(2),
        90 => Some(2),
        40 => Some(3),
        50 => Some(4),
        60 => Some(5),
        80 => Some(6),
        _ => None
    }
}

pub async fn download_labels(client: &Client, year: u32, anchor: &AnchorRef, out_path: &Path) -> Result<()> {
    if year != 2020 && year != 2021 {
        return Ok(());
    }
    let version = if year == 2020 { "v100" } else { "v200" };

    let crs_epsg = anchor.epsg;
    let is_north = crs_epsg >= 32600 && crs_epsg < 32700;
    let is_south = crs_epsg >= 32700 && crs_epsg < 32800;
    if !is_north && !is_south {
        anyhow::bail!("Unsupported EPSG {} for UTM-to-WGS84 conversion", crs_epsg);
    }
    let zone = (crs_epsg % 100) as u8;
    let hemisphere = if is_north { 'N' } else { 'S' };

    let t = &anchor.geo_transform;
    let cx = t.origin_x + (anchor.width as f64 / 2.0) * t.pixel_size_x;
    let cy = t.origin_y - (anchor.height as f64 / 2.0) * t.pixel_size_y;
    
    let (lat_center, lon_center) = utm::wsg84_utm_to_lat_lon(cx, cy, zone, hemisphere)
        .map_err(|e| anyhow::anyhow!("UTM->WGS84 center conversion failed: {:?}", e))?;

    let lat_tile = (lat_center / 3.0).floor() as i32 * 3;
    let lon_tile = (lon_center / 3.0).floor() as i32 * 3;
    let ns = if lat_tile >= 0 { "N" } else { "S" };
    let ew = if lon_tile >= 0 { "E" } else { "W" };
    let tile = format!("{}{:02}{}{:03}", ns, lat_tile.abs(), ew, lon_tile.abs());

    let filename = format!("ESA_WorldCover_10m_{}_{}_{}_Map.tif", year, version, tile);
    let url = format!("https://esa-worldcover.s3.eu-central-1.amazonaws.com/{}/{}/map/{}", version, year, filename);

    println!("  Labels {}: Fetching {}", year, filename);

    let meta = read_cog_meta(client, &url).await?;
    let src_gt = crate::reproject::GeoTransform::from_cog(&meta.pixel_scale, &meta.tiepoint);

    let mut min_lat = 90.0f64;
    let mut max_lat = -90.0f64;
    let mut min_lon = 180.0f64;
    let mut max_lon = -180.0f64;

    let corners = [
        (t.origin_x, t.origin_y),
        (t.origin_x + anchor.width as f64 * t.pixel_size_x, t.origin_y),
        (t.origin_x, t.origin_y - anchor.height as f64 * t.pixel_size_y),
        (t.origin_x + anchor.width as f64 * t.pixel_size_x, t.origin_y - anchor.height as f64 * t.pixel_size_y),
    ];

    for &(gx, gy) in &corners {
        let (lat, lon) = utm::wsg84_utm_to_lat_lon(gx, gy, zone, hemisphere)
            .map_err(|e| anyhow::anyhow!("UTM->WGS84 corner conversion failed: {:?}", e))?;
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
        min_lon = min_lon.min(lon);
        max_lon = max_lon.max(lon);
    }

    let (sx0_f, sy0_f) = src_gt.geo_to_pixel(min_lon, max_lat); 
    let (sx1_f, sy1_f) = src_gt.geo_to_pixel(max_lon, min_lat);

    let pad = 2;
    let src_bbox = PixelBbox {
        x0: (sx0_f.min(sx1_f).floor() as i64 - pad as i64).max(0) as u32,
        y0: (sy0_f.min(sy1_f).floor() as i64 - pad as i64).max(0) as u32,
        x1: ((sx0_f.max(sx1_f).ceil() as u32 + pad as u32).min(meta.width)).max(0),
        y1: ((sy0_f.max(sy1_f).ceil() as u32 + pad as u32).min(meta.height)).max(0),
    };

    let raw_pixels = read_cog_region(client, &url, &meta, src_bbox.clone()).await?;
    let raw_w = (src_bbox.x1 - src_bbox.x0) as usize;
    let raw_h = (src_bbox.y1 - src_bbox.y0) as usize;

    let grid_px: usize = 10;
    let nc = anchor.width / grid_px;
    let nr = anchor.height / grid_px;
    let n_cells = nc * nr;
    
    let mut cell_counts = vec![[0u32; N_CLASSES]; n_cells];
    
    use rayon::prelude::*;
    cell_counts.par_iter_mut().enumerate().for_each(|(cell_id, counts)| {
        let ci = cell_id % nc;
        let ri = cell_id / nc;
                
        for dy in 0..grid_px {
            for dx in 0..grid_px {
                let px = ci * grid_px + dx;
                let py = ri * grid_px + dy;
                
                let (easting, northing) = t.pixel_to_geo(px as f64 + 0.5, py as f64 + 0.5);
                let Ok((lat, lon)) = utm::wsg84_utm_to_lat_lon(easting, northing, zone, hemisphere) else {
                    continue; // skip pixels with invalid UTM coordinates
                };
                
                let (sx_f, sy_f) = src_gt.geo_to_pixel(lon, lat);
                let ix = sx_f.floor() as isize - src_bbox.x0 as isize;
                let iy = sy_f.floor() as isize - src_bbox.y0 as isize;
                
                if ix >= 0 && iy >= 0 && ix < raw_w as isize && iy < raw_h as isize {
                    let val = raw_pixels[iy as usize * raw_w + ix as usize];
                    if val.is_finite() && val > 0.0 {
                        if let Some(cidx) = map_wc_class(val.round() as u8) {
                            counts[cidx] += 1;
                        }
                    }
                }
            }
        }
    });

    let mut result_map = serde_json::Map::with_capacity(n_cells);
    let total_px = (grid_px * grid_px) as f32;
    for cell_id in 0..n_cells {
        let mut cell_props = serde_json::Map::with_capacity(N_CLASSES);
        for i in 0..N_CLASSES {
            let prop = (cell_counts[cell_id][i] as f32) / total_px;
            let rounded = (prop * 10000.0).round() / 10000.0;
            cell_props.insert(CLASS_NAMES[i].to_string(), json!(rounded));
        }
        result_map.insert(cell_id.to_string(), serde_json::Value::Object(cell_props));
    }
    
    let contents = serde_json::to_string(&serde_json::Value::Object(result_map))?;
    std::fs::write(out_path, contents)?;

    println!("  Wrote labels json {} ({} cells)", out_path.display(), n_cells);

    Ok(())
}
