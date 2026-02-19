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

/// Resample source raster to target grid using bilinear interpolation.
///
/// Both source and target must share the same CRS. If CRS differs and the
/// difference is just a UTM zone shift, use `resample_with_offset` instead.
///
/// Returns a flat f32 buffer of size (dst_h × dst_w), with NaN for out-of-bounds.
pub fn resample_bilinear(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    src_gt: &GeoTransform,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
) -> Vec<f32> {
    let mut output = vec![f32::NAN; dst_h * dst_w];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Target pixel center → geo coordinate
            let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);

            // Geo → source pixel (fractional)
            let (sx, sy) = src_gt.geo_to_pixel(gx, gy);

            // Source pixel center offset (0.5 compensation)
            let sx = sx - 0.5;
            let sy = sy - 0.5;

            // Bounds check
            if sx < -0.5 || sy < -0.5 || sx >= src_w as f64 - 0.5 || sy >= src_h as f64 - 0.5 {
                continue;
            }

            // Bilinear interpolation
            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;

            let sample = |r: isize, c: isize| -> f64 {
                if r < 0 || c < 0 || r >= src_h as isize || c >= src_w as isize {
                    return f64::NAN;
                }
                let v = src[r as usize * src_w + c as usize];
                if v.is_finite() { v as f64 } else { f64::NAN }
            };

            let v00 = sample(y0, x0);
            let v10 = sample(y0, x1);
            let v01 = sample(y1, x0);
            let v11 = sample(y1, x1);

            // NaN-aware bilinear: if any neighbor is NaN, use nearest valid
            let val = if v00.is_finite() && v10.is_finite() && v01.is_finite() && v11.is_finite() {
                let top = v00 * (1.0 - fx) + v10 * fx;
                let bot = v01 * (1.0 - fx) + v11 * fx;
                top * (1.0 - fy) + bot * fy
            } else {
                // Fallback: weighted average of finite values
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
                if wsum > 0.0 { vsum / wsum } else { f64::NAN }
            };

            output[dy * dst_w + dx] = val as f32;
        }
    }

    output
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

    output.par_chunks_mut(dst_w)
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
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let sample = |r: isize, c: isize| -> f64 {
                    if r < 0 || c < 0 || r >= src_h as isize || c >= src_w as isize {
                        return f64::NAN;
                    }
                    let v = src[r as usize * src_w + c as usize];
                    if v.is_finite() { v as f64 } else { f64::NAN }
                };

                let v00 = sample(y0, x0);
                let v10 = sample(y0, x1);
                let v01 = sample(y1, x0);
                let v11 = sample(y1, x1);

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
                    if wsum > 0.0 { vsum / wsum } else { f64::NAN }
                };

                row[dx] = val as f32;
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

    output.par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            for dx in 0..dst_w {
                // Target pixel center → geo coordinate
                let (gx, gy) = dst_gt.pixel_to_geo(dx as f64 + 0.5, dy as f64 + 0.5);

                // Geo → source pixel (fractional, corner-based)
                let (sx, sy) = src_gt.geo_to_pixel(gx, gy);

                // Convert to center-index coords and round to nearest
                let ix = (sx - 0.5 + 0.5).floor() as isize; // equivalent to sx.floor() but clearer intent
                let iy = (sy - 0.5 + 0.5).floor() as isize;

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
