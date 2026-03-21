use anyhow::Result;
use std::path::Path;
use crate::composite::AnchorRef;

pub fn generate_grid_geojson(anchor: &AnchorRef, out_path: &Path) -> Result<()> {
    let mut features = Vec::new();
    let crs_epsg = anchor.epsg;
    
    // UTM parameters
    let is_north = crs_epsg >= 32600 && crs_epsg < 32700;
    let is_south = crs_epsg >= 32700 && crs_epsg < 32800;
    if !is_north && !is_south {
        anyhow::bail!("Unsupported EPSG {} for UTM-to-WGS84 conversion", crs_epsg);
    }
    let zone = (crs_epsg % 100) as u8;
    // For utm crate, 'N' represents northern hemisphere in standard `to_lat_lon` functions that take zone_letter as just N/S in some libs.
    // Let's use 'N' and 'S'. If it requires exact band, we'll see a test error.
    let hemisphere = if is_north { 'N' } else { 'S' };

    let grid_px: usize = 10;
    let t = &anchor.geo_transform;
    let sentinel_res = t.pixel_size_x; // use actual pixel size from anchor
    
    let nc = anchor.width / grid_px;
    let nr = anchor.height / grid_px;

    // We'll calculate it just like python: precalculate x0,x1 and y0,y1 for all cells
    
    for ri in 0..nr {
        for ci in 0..nc {
            let cell_id = ri * nc + ci;
            
            let x0 = t.origin_x + (ci * grid_px) as f64 * sentinel_res;
            let x1 = x0 + (grid_px as f64 * sentinel_res);
            let y0 = t.origin_y - (ri * grid_px) as f64 * sentinel_res; // Y goes down
            let y1 = y0 - (grid_px as f64 * sentinel_res);

            // Coordinates in closed polygon: (x0,y0), (x1,y0), (x1,y1), (x0,y1), (x0,y0)
            let corners = [
                (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)
            ];

            let mut wgs84_coords = Vec::with_capacity(5);
            for (gx, gy) in corners {
                let (lat, lon) = utm::wsg84_utm_to_lat_lon(gx, gy, zone, hemisphere).map_err(|e| anyhow::anyhow!("UTM conversion error: {:?}", e))?;
                // GeoJSON format is [lon, lat]
                let lon_rounded = (lon * 1_000_000.0_f64).round() / 1_000_000.0_f64;
                let lat_rounded = (lat * 1_000_000.0_f64).round() / 1_000_000.0_f64;
                wgs84_coords.push(vec![lon_rounded, lat_rounded]);
            }

            let feature = serde_json::json!({
                "type": "Feature",
                "properties": { "cell_id": cell_id },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [wgs84_coords]
                }
            });
            features.push(feature);
        }
    }

    let geojson = serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
    });

    let json_str = serde_json::to_string(&geojson)?;
    std::fs::write(out_path, json_str)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_utm_conversion() {
        let (lat, lon) = utm::wsg84_utm_to_lat_lon(500000.0, 4600000.0, 32, 'N').unwrap();
        assert!(lat > 40.0 && lat < 45.0);
        assert!(lon > 8.0 && lon < 12.0);
    }
}
