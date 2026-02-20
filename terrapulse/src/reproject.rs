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
