//! Bilinear resampling for reprojecting raster data between grids.
//!
//! Handles the common case where source and target share the same CRS
//! (just different pixel grids), which covers ~99% of Sentinel-2 over Europe.

/// Affine geo-transform (matches GDAL/rasterio convention).
///
/// pixel(x, y) → geo(x, y):
///   geo_x = origin_x + pixel_x * pixel_size_x
///   geo_y = origin_y - pixel_y * pixel_size_y  (note: Y is flipped)
#[derive(Debug, Clone, Copy)]
pub struct GeoTransform {
    pub origin_x: f64,
    pub origin_y: f64,
    pub pixel_size_x: f64,
    pub pixel_size_y: f64, // positive value; geo_y decreases as pixel_y increases
}

impl GeoTransform {
    /// Pixel → geo coordinate.
    #[inline]
    pub fn pixel_to_geo(&self, px: f64, py: f64) -> (f64, f64) {
        (
            self.origin_x + px * self.pixel_size_x,
            self.origin_y - py * self.pixel_size_y,
        )
    }

    /// Geo → pixel coordinate (fractional).
    #[inline]
    pub fn geo_to_pixel(&self, gx: f64, gy: f64) -> (f64, f64) {
        (
            (gx - self.origin_x) / self.pixel_size_x,
            (self.origin_y - gy) / self.pixel_size_y,
        )
    }

    /// Construct from COG metadata (pixel_scale + tiepoint).
    pub fn from_cog(pixel_scale: &[f64; 3], tiepoint: &[f64; 6]) -> Self {
        Self {
            origin_x: tiepoint[3] - tiepoint[0] * pixel_scale[0],
            origin_y: tiepoint[4] + tiepoint[1] * pixel_scale[1],
            pixel_size_x: pixel_scale[0],
            pixel_size_y: pixel_scale[1],
        }
    }
}

/// NaN-aware bilinear interpolation from four corner samples.
///
/// `fx`, `fy` are fractional positions ∈ [0, 1) within the pixel cell.
/// If all four corners are finite, standard bilinear blending is used.
/// Otherwise, a weighted average of the finite neighbours is returned.
/// Returns `f32::NAN` if no neighbours are finite.
#[inline]
pub fn bilinear_interp(v00: f64, v10: f64, v01: f64, v11: f64, fx: f64, fy: f64) -> f32 {
    let val = if v00.is_finite() && v10.is_finite() && v01.is_finite() && v11.is_finite() {
        let top = v00 * (1.0 - fx) + v10 * fx;
        let bot = v01 * (1.0 - fx) + v11 * fx;
        top * (1.0 - fy) + bot * fy
    } else {
        let weights = [
            ((1.0 - fx) * (1.0 - fy), v00),
            (fx * (1.0 - fy), v10),
            ((1.0 - fx) * fy, v01),
            (fx * fy, v11),
        ];
        let mut wsum = 0.0;
        let mut vsum = 0.0;
        for &(w, v) in &weights {
            if v.is_finite() {
                wsum += w;
                vsum += w * v;
            }
        }
        if wsum > 0.0 {
            vsum / wsum
        } else {
            f64::NAN
        }
    };
    val as f32
}

/// Resample source raster using parallel row processing (for large rasters).
pub fn resample_bilinear_par(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    src_gt: &GeoTransform,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
) -> Vec<f32> {
    use rayon::prelude::*;

    let mut output = vec![f32::NAN; dst_h * dst_w];

    output
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            for dx in 0..dst_w {
                let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);
                let (sx, sy) = src_gt.geo_to_pixel(gx, gy);
                let sx = sx - 0.5;
                let sy = sy - 0.5;

                if sx < -0.5 || sy < -0.5 || sx >= src_w as f64 - 0.5 || sy >= src_h as f64 - 0.5 {
                    continue;
                }

                let x0 = sx.floor() as isize;
                let y0 = sy.floor() as isize;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let sample = |r: isize, c: isize| -> f64 {
                    if r < 0 || c < 0 || r >= src_h as isize || c >= src_w as isize {
                        return f64::NAN;
                    }
                    let v = src[r as usize * src_w + c as usize];
                    if v.is_finite() {
                        v as f64
                    } else {
                        f64::NAN
                    }
                };

                row[dx] = bilinear_interp(
                    sample(y0, x0),
                    sample(y0, x0 + 1),
                    sample(y0 + 1, x0),
                    sample(y0 + 1, x0 + 1),
                    fx,
                    fy,
                );
            }
        });

    output
}

/// Resample source raster using nearest-neighbor (parallel).
///
/// Use this for categorical data (e.g. SCL class masks) where bilinear
/// interpolation would blend class IDs into meaningless values.
pub fn resample_nearest_par(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    src_gt: &GeoTransform,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
) -> Vec<f32> {
    use rayon::prelude::*;

    let mut output = vec![f32::NAN; dst_h * dst_w];

    output
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            for dx in 0..dst_w {
                // Target pixel center → geo coordinate
                let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);

                // Geo → source pixel (fractional, corner-based)
                let (sx, sy) = src_gt.geo_to_pixel(gx, gy);

                // Round to nearest source pixel
                let ix = sx.floor() as isize;
                let iy = sy.floor() as isize;

                if ix < 0 || iy < 0 || ix >= src_w as isize || iy >= src_h as isize {
                    continue;
                }
                let v = src[iy as usize * src_w + ix as usize];
                if v.is_finite() {
                    row[dx] = v;
                }
            }
        });

    output
}

// ── UTM ↔ Geographic coordinate conversion ──
//
// Standard UTM projection using the transverse Mercator formulas.
// Reference: Snyder, "Map Projections — A Working Manual" (USGS Prof. Paper 1395)

const WGS84_A: f64 = 6_378_137.0; // semi-major axis
const WGS84_F: f64 = 1.0 / 298.257_223_563; // flattening
const UTM_K0: f64 = 0.9996; // scale factor
const UTM_FE: f64 = 500_000.0; // false easting

/// Extract UTM zone number and hemisphere from an EPSG code.
/// Returns (zone, is_north). Covers EPSG 326xx (north) and 327xx (south).
#[inline]
pub fn epsg_to_zone(epsg: u32) -> (u32, bool) {
    if epsg >= 32601 && epsg <= 32660 {
        (epsg - 32600, true)
    } else if epsg >= 32701 && epsg <= 32760 {
        (epsg - 32700, false)
    } else {
        // Fallback: assume zone 32 north (Central Europe)
        (32, true)
    }
}

/// Central meridian for a UTM zone.
#[inline]
fn zone_central_meridian(zone: u32) -> f64 {
    (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0
}

/// Convert UTM (easting, northing) to geographic (longitude, latitude) in degrees.
pub fn utm_to_geographic(easting: f64, northing: f64, zone: u32, is_north: bool) -> (f64, f64) {
    let e = WGS84_F * (2.0 - WGS84_F); // first eccentricity squared
    let e1sq = e / (1.0 - e);
    let _n_val = WGS84_A / (1.0 - e).sqrt();

    let fn_val = if is_north { 0.0 } else { 10_000_000.0 };
    let cm = zone_central_meridian(zone).to_radians();

    let x = easting - UTM_FE;
    let y = northing - fn_val;

    let m = y / UTM_K0;

    // Footpoint latitude by iteration (Bowring's method)
    let mu = m / (WGS84_A * (1.0 - e / 4.0 - 3.0 * e * e / 64.0 - 5.0 * e * e * e / 256.0));
    let e1 = (1.0 - (1.0 - e).sqrt()) / (1.0 + (1.0 - e).sqrt());

    let fp_lat = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let c1 = e1sq * fp_lat.cos().powi(2);
    let t1 = fp_lat.tan().powi(2);
    let r1 = WGS84_A * (1.0 - e) / (1.0 - e * fp_lat.sin().powi(2)).powf(1.5);
    let n1 = WGS84_A / (1.0 - e * fp_lat.sin().powi(2)).sqrt();
    let d = x / (n1 * UTM_K0);

    let lat = fp_lat
        - (n1 * fp_lat.tan() / r1)
            * (d * d / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * e1sq) * d.powi(4) / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1
                    - 252.0 * e1sq
                    - 3.0 * c1 * c1)
                    * d.powi(6)
                    / 720.0);

    let lon = cm
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * e1sq + 24.0 * t1 * t1)
                * d.powi(5)
                / 120.0)
            / fp_lat.cos();

    (lon.to_degrees(), lat.to_degrees())
}

/// Convert geographic (longitude, latitude) in degrees to UTM (easting, northing).
pub fn geographic_to_utm(lon_deg: f64, lat_deg: f64, zone: u32, is_north: bool) -> (f64, f64) {
    let e = WGS84_F * (2.0 - WGS84_F);
    let e1sq = e / (1.0 - e);

    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let cm = zone_central_meridian(zone).to_radians();

    let n_val = WGS84_A / (1.0 - e * lat.sin().powi(2)).sqrt();
    let t = lat.tan().powi(2);
    let c = e1sq * lat.cos().powi(2);
    let a_val = (lon - cm) * lat.cos();

    let m = WGS84_A
        * ((1.0 - e / 4.0 - 3.0 * e * e / 64.0 - 5.0 * e * e * e / 256.0) * lat
            - (3.0 * e / 8.0 + 3.0 * e * e / 32.0 + 45.0 * e * e * e / 1024.0)
                * (2.0 * lat).sin()
            + (15.0 * e * e / 256.0 + 45.0 * e * e * e / 1024.0) * (4.0 * lat).sin()
            - (35.0 * e * e * e / 3072.0) * (6.0 * lat).sin());

    let easting = UTM_FE
        + UTM_K0
            * n_val
            * (a_val
                + (1.0 - t + c) * a_val.powi(3) / 6.0
                + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * e1sq) * a_val.powi(5) / 120.0);

    let fn_val = if is_north { 0.0 } else { 10_000_000.0 };
    let northing = fn_val
        + UTM_K0
            * (m
                + n_val * lat.tan()
                    * (a_val * a_val / 2.0
                        + (5.0 - t + 9.0 * c + 4.0 * c * c) * a_val.powi(4) / 24.0
                        + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * e1sq) * a_val.powi(6)
                            / 720.0));

    (easting, northing)
}

/// Convert UTM coordinates from one zone to another.
#[inline]
pub fn utm_to_utm(
    easting: f64,
    northing: f64,
    src_zone: u32,
    src_north: bool,
    dst_zone: u32,
    dst_north: bool,
) -> (f64, f64) {
    let (lon, lat) = utm_to_geographic(easting, northing, src_zone, src_north);
    geographic_to_utm(lon, lat, dst_zone, dst_north)
}

/// Resample with cross-CRS support (bilinear). Destination and source may be
/// in different UTM zones. Transforms dst geo → lat/lon → src geo per pixel.
pub fn resample_bilinear_cross_crs(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    src_gt: &GeoTransform,
    src_epsg: u32,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
    dst_epsg: u32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let (dst_zone, dst_north) = epsg_to_zone(dst_epsg);
    let (src_zone, src_north) = epsg_to_zone(src_epsg);

    let mut output = vec![f32::NAN; dst_h * dst_w];

    output
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            for dx in 0..dst_w {
                // dst pixel → dst UTM geo
                let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);
                // dst UTM → src UTM
                let (sgx, sgy) = utm_to_utm(gx, gy, dst_zone, dst_north, src_zone, src_north);
                // src UTM geo → src pixel
                let (sx, sy) = src_gt.geo_to_pixel(sgx, sgy);
                let sx = sx - 0.5;
                let sy = sy - 0.5;

                if sx < -0.5 || sy < -0.5 || sx >= src_w as f64 - 0.5 || sy >= src_h as f64 - 0.5 {
                    continue;
                }

                let x0 = sx.floor() as isize;
                let y0 = sy.floor() as isize;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let sample = |r: isize, c: isize| -> f64 {
                    if r < 0 || c < 0 || r >= src_h as isize || c >= src_w as isize {
                        return f64::NAN;
                    }
                    let v = src[r as usize * src_w + c as usize];
                    if v.is_finite() { v as f64 } else { f64::NAN }
                };

                row[dx] = bilinear_interp(
                    sample(y0, x0),
                    sample(y0, x0 + 1),
                    sample(y0 + 1, x0),
                    sample(y0 + 1, x0 + 1),
                    fx,
                    fy,
                );
            }
        });

    output
}

/// Resample with cross-CRS support (nearest-neighbor).
pub fn resample_nearest_cross_crs(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    src_gt: &GeoTransform,
    src_epsg: u32,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
    dst_epsg: u32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let (dst_zone, dst_north) = epsg_to_zone(dst_epsg);
    let (src_zone, src_north) = epsg_to_zone(src_epsg);

    let mut output = vec![f32::NAN; dst_h * dst_w];

    output
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            for dx in 0..dst_w {
                let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);
                let (sgx, sgy) = utm_to_utm(gx, gy, dst_zone, dst_north, src_zone, src_north);
                let (sx, sy) = src_gt.geo_to_pixel(sgx, sgy);

                let ix = sx.floor() as isize;
                let iy = sy.floor() as isize;

                if ix < 0 || iy < 0 || ix >= src_w as isize || iy >= src_h as isize {
                    continue;
                }
                let v = src[iy as usize * src_w + ix as usize];
                if v.is_finite() {
                    row[dx] = v;
                }
            }
        });

    output
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_transform() {
        let gt = GeoTransform {
            origin_x: 100.0,
            origin_y: 50.0,
            pixel_size_x: 10.0,
            pixel_size_y: 10.0,
        };

        // pixel -> geo
        // (0, 0) -> (100, 50)
        let (gx, gy) = gt.pixel_to_geo(0.0, 0.0);
        assert_eq!(gx, 100.0);
        assert_eq!(gy, 50.0);

        // (1, 1) -> (110, 40) -- note Y decreases
        let (gx, gy) = gt.pixel_to_geo(1.0, 1.0);
        assert_eq!(gx, 110.0);
        assert_eq!(gy, 40.0);

        // fractional
        let (gx, gy) = gt.pixel_to_geo(0.5, 0.5);
        assert_eq!(gx, 105.0);
        assert_eq!(gy, 45.0);

        // geo -> pixel
        let (px, py) = gt.geo_to_pixel(105.0, 45.0);
        assert_eq!(px, 0.5);
        assert_eq!(py, 0.5);
    }

    #[test]
    fn test_geo_transform_from_cog() {
        let pixel_scale = [10.0, 10.0, 0.0];
        // tiepoint: [I, J, K, X, Y, Z]
        let tiepoint = [0.0, 0.0, 0.0, 100.0, 50.0, 0.0];
        let gt = GeoTransform::from_cog(&pixel_scale, &tiepoint);
        
        assert_eq!(gt.origin_x, 100.0);
        assert_eq!(gt.origin_y, 50.0);
        assert_eq!(gt.pixel_size_x, 10.0);
        assert_eq!(gt.pixel_size_y, 10.0);
    }

    #[test]
    fn test_bilinear_interp() {
        // all finite
        let act = bilinear_interp(10.0, 20.0, 30.0, 40.0, 0.5, 0.5);
        assert_eq!(act, 25.0); // center of 10,20,30,40 is 25

        let act = bilinear_interp(10.0, 20.0, 30.0, 40.0, 0.0, 0.0);
        assert_eq!(act, 10.0); // exact top-left corner

        let act = bilinear_interp(10.0, 20.0, 30.0, 40.0, 1.0, 1.0);
        assert_eq!(act, 40.0); // exact bottom-right corner

        // with NaNs (fallback to average of finite weights)
        // Only v00 is finite, w00 is max at fx=0, fy=0
        let act = bilinear_interp(10.0, f64::NAN, f64::NAN, f64::NAN, 0.0, 0.0);
        assert_eq!(act, 10.0);

        // 50/50 mix between v00 and v10 because v01 and v11 are NaN
        // weights: w00=(1-0.5)*(1-0) = 0.5, w10=0.5*(1-0) = 0.5
        let act = bilinear_interp(10.0, 20.0, f64::NAN, f64::NAN, 0.5, 0.0);
        assert_eq!(act, 15.0);

        // All NaN
        let act = bilinear_interp(f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.5, 0.5);
        assert!(act.is_nan());
    }
}
