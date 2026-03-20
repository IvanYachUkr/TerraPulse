//! Scene compositing: download all bands from multiple scenes, cloud-mask,
//! and produce a nanmedian composite. Pure Rust replacement for composite.py.

use anyhow::{Context, Result};
use rayon::prelude::*;
use reqwest::Client;
use std::collections::HashMap;
use std::path::Path;

use crate::cog::{self, PixelBbox};
use crate::reproject::{self, GeoTransform};
use crate::stac::StacItem;
// ── Constants matching composite.py ──

/// SCL classes to exclude (cloud, shadow, saturated, etc.)
/// Matches the reference algorithm:
///   1 = SATURATED_DEFECTIVE
///   3 = CLOUD_SHADOW
///   7 = CLOUD_LOW_PROBA / UNCLASSIFIED  (haze contaminates Q1 composites)
///   8 = CLOUD_MEDIUM_PROBA
///   9 = CLOUD_HIGH_PROBA
///  10 = THIN_CIRRUS
/// Note: SCL 0 (no_data) is handled implicitly by the is_finite() check.
///       SCL 11 (snow) is kept — rare in our study area and provides valid data.
const SCL_EXCLUDE: [u8; 6] = [1, 3, 7, 8, 9, 10];

/// Spectral bands to download (same order as composite.py)
const SPECTRAL_BANDS: [&str; 10] = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
];

const NODATA_VAL: f32 = -9999.0;

/// Anchor reference metadata (target grid definition).
#[derive(Clone)]
pub struct AnchorRef {
    pub width: usize,
    pub height: usize,
    pub geo_transform: GeoTransform,
    pub epsg: u32,
}

impl AnchorRef {
    /// Read anchor metadata from a local GeoTIFF.
    pub fn from_tif(path: &Path) -> Result<Self> {
        let meta = cog::read_local_tif_meta(path)?;
        Ok(Self {
            width: meta.width as usize,
            height: meta.height as usize,
            geo_transform: GeoTransform::from_cog(&meta.pixel_scale, &meta.tiepoint),
            epsg: meta.epsg,
        })
    }
}

/// Per-scene data: 10 spectral bands + cloud mask, all resampled to target grid.
struct SceneData {
    /// [10][height * width] spectral bands
    bands: Vec<Vec<f32>>,
    /// [height * width] cloud mask (true = valid, false = cloudy/excluded)
    valid_mask: Vec<bool>,
}

/// Download, reproject, cloud-mask, and composite all scenes for one season.
///
/// Produces the same 11-band output as composite.py:
/// bands 0-9 = spectral median composite, band 10 = valid fraction.
pub async fn download_and_composite(
    client: &Client,
    items: &[StacItem],
    signed_urls: &[HashMap<String, String>],
    anchor: &AnchorRef,
    output_path: &Path,
    year: u32,
) -> Result<()> {
    let n_scenes = items.len();
    let n_bands = SPECTRAL_BANDS.len();
    let n_pixels = anchor.width * anchor.height;

    eprintln!(
        "  Compositing {n_scenes} scenes -> {}x{} ...",
        anchor.width, anchor.height
    );
    eprintln!("    Downloading {n_scenes} scenes (all parallel)...");

    // Download all scenes concurrently, with per-scene retry
    let scene_futures: Vec<_> = signed_urls
        .iter()
        .enumerate()
        .map(|(si, band_urls)| {
            let client = client.clone();
            let urls = band_urls.clone();
            let anchor_w = anchor.width;
            let anchor_h = anchor.height;
            let anchor_gt = anchor.geo_transform;
            let anchor_epsg = anchor.epsg;
            async move {
                let max_retries = 2u32;
                let mut last_err = String::new();
                for attempt in 0..=max_retries {
                    if attempt > 0 {
                        eprintln!("    Scene {}: retry {attempt}/{max_retries}...", si + 1);
                        tokio::time::sleep(std::time::Duration::from_secs(3 * attempt as u64))
                            .await;
                    }
                    match download_one_scene(
                        &client,
                        &urls,
                        anchor_w,
                        anchor_h,
                        &anchor_gt,
                        anchor_epsg,
                    )
                    .await
                    {
                        Ok(data) => {
                            if attempt > 0 {
                                eprintln!("    Scene {}: OK (after {attempt} retries)", si + 1);
                            } else {
                                eprintln!("    Scene {}: OK", si + 1);
                            }
                            return Ok(data);
                        }
                        Err(e) => {
                            last_err = format!("{e:#}");
                            if attempt == max_retries {
                                eprintln!(
                                    "    Scene {}: FAILED after {max_retries} retries - {last_err}",
                                    si + 1
                                );
                            }
                        }
                    }
                }
                Err(anyhow::anyhow!("Scene {} failed: {}", si + 1, last_err))
            }
        })
        .collect();

    // Use buffer_unordered to limit concurrent scene downloads and avoid OOM
    use futures::stream::{self, StreamExt};
    let results: Vec<_> = stream::iter(scene_futures)
        .buffer_unordered(6)
        .collect()
        .await;

    // Collect successful scenes
    let mut scenes: Vec<SceneData> = Vec::new();
    let mut n_ok = 0;
    let mut n_fail = 0;
    for r in results {
        match r {
            Ok(sd) => {
                scenes.push(sd);
                n_ok += 1;
            }
            Err(_) => {
                n_fail += 1;
            }
        }
    }
    eprintln!("    {n_ok}/{n_scenes} scenes OK, {n_fail} failed");

    if scenes.is_empty() {
        anyhow::bail!("No scenes downloaded successfully");
    }

    // ESA Processing Baseline 04.00 (effective Jan 25 2022) added a +1000
    // BOA_ADD_OFFSET to Sentinel-2 L2A surface reflectance values.
    // Detect per-scene: if B02 median > 900, subtract 1000 from spectral bands.
    // This handles the 2022 transitional year where old/new baselines coexist.
    if year >= 2022 {
        let boa_offset = 1000.0f32;
        let mut n_corrected = 0;
        for scene in &mut scenes {
            // B02 is band index 0 (first in SPECTRAL_BANDS)
            let b02 = &scene.bands[0];
            let mut valid_vals: Vec<f32> = b02
                .iter()
                .zip(scene.valid_mask.iter())
                .filter(|(&v, &m)| m && v.is_finite() && v > 0.0)
                .map(|(&v, _)| v)
                .collect();
            if valid_vals.is_empty() {
                continue;
            }
            valid_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let median = valid_vals[valid_vals.len() / 2];

            if median > 900.0 {
                // This scene has the BOA offset — subtract from all spectral bands
                for bi in 0..n_bands {
                    for px in 0..n_pixels {
                        let v = scene.bands[bi][px];
                        if v.is_finite() && v > 0.0 {
                            scene.bands[bi][px] = (v - boa_offset).max(1.0);
                        }
                    }
                }
                n_corrected += 1;
            }
        }
        if n_corrected > 0 {
            eprintln!(
                "    BOA_ADD_OFFSET: corrected {n_corrected}/{} scenes (year {year})",
                scenes.len()
            );
        }
    }

    // Compute nanmedian composite using Rayon
    let mut composite = vec![NODATA_VAL; n_bands * n_pixels];
    let mut valid_fraction = vec![0.0f32; n_pixels];

    // Pre-allocate indices to distribute workload
    let pixel_indices: Vec<usize> = (0..n_pixels).collect();

    // Compute pixel values in parallel (embarrassingly parallel over all cores)
    let results: Vec<(Vec<f32>, f32)> = pixel_indices
        .into_par_iter()
        .map(|px| {
            let mut n_valid = 0u32;
            for scene in &scenes {
                if scene.valid_mask[px] {
                    n_valid += 1;
                }
            }
            let valid_frac = n_valid as f32 / scenes.len() as f32;

            if n_valid == 0 {
                return (Vec::new(), valid_frac);
            }

            let mut medians = Vec::with_capacity(n_bands);
            for bi in 0..n_bands {
                let mut vals: Vec<f32> = Vec::with_capacity(n_valid as usize);
                for scene in &scenes {
                    if scene.valid_mask[px] {
                        let v = scene.bands[bi][px];
                        if v.is_finite() {
                            vals.push(v);
                        }
                    }
                }
                if !vals.is_empty() {
                    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    // TARGET ALGORITHM: Take the value of the first quartile (25th percentile)
                    // instead of the median (50th percentile).
                    let q1_idx = vals.len() / 4;
                    let quartile = vals[q1_idx];
                    medians.push(quartile);
                } else {
                    medians.push(NODATA_VAL);
                }
            }
            (medians, valid_frac)
        })
        .collect();

    // Write back computed values to continuous slices
    for (px, (medians, frac)) in results.into_iter().enumerate() {
        valid_fraction[px] = frac;
        if !medians.is_empty() {
            for bi in 0..n_bands {
                composite[bi * n_pixels + px] = medians[bi];
            }
        }
    }

    // Write output GeoTIFF (pixel-interleaved, 11 bands: 10 spectral + valid_fraction)
    write_composite_tif(output_path, anchor, &composite, &valid_fraction)?;

    Ok(())
}

/// Download and resample one scene's 10 spectral bands + SCL.
async fn download_one_scene(
    client: &Client,
    band_urls: &HashMap<String, String>,
    dst_w: usize,
    dst_h: usize,
    dst_gt: &GeoTransform,
    dst_epsg: u32,
) -> Result<SceneData> {
    let n_bands = SPECTRAL_BANDS.len();

    // First, read one band's metadata to get source dimensions and transform
    let first_band_url = band_urls.get("B02").context("Missing B02 in signed URLs")?;
    let src_meta = cog::read_cog_meta(client, first_band_url)
        .await
        .context("Failed to read COG metadata")?;

    let src_gt = GeoTransform::from_cog(&src_meta.pixel_scale, &src_meta.tiepoint);

    // Check CRS — if different, we'll use cross-CRS resampling instead of bailing
    let epsg_mismatch = src_meta.epsg != 0 && dst_epsg != 0 && src_meta.epsg != dst_epsg;
    if epsg_mismatch {
        eprintln!(
            "    (cross-CRS: source EPSG:{} → target EPSG:{})",
            src_meta.epsg, dst_epsg
        );
    }

    // Calculate which source pixels we need (target bbox in source pixel coords)
    // When CRS differs, transform target corners to source CRS first
    let (tl_gx, tl_gy) = dst_gt.pixel_to_geo(0.0, 0.0);
    let (br_gx, br_gy) = dst_gt.pixel_to_geo(dst_w as f64, dst_h as f64);

    let (tl_sx, tl_sy, br_sx, br_sy) = if epsg_mismatch {
        let (dst_zone, dst_north) = reproject::epsg_to_zone(dst_epsg);
        let (src_zone, src_north) = reproject::epsg_to_zone(src_meta.epsg);
        let (tl_e, tl_n) = reproject::utm_to_utm(tl_gx, tl_gy, dst_zone, dst_north, src_zone, src_north);
        let (br_e, br_n) = reproject::utm_to_utm(br_gx, br_gy, dst_zone, dst_north, src_zone, src_north);
        let (tl_sx, tl_sy) = src_gt.geo_to_pixel(tl_e, tl_n);
        let (br_sx, br_sy) = src_gt.geo_to_pixel(br_e, br_n);
        (tl_sx, tl_sy, br_sx, br_sy)
    } else {
        let (tl_sx, tl_sy) = src_gt.geo_to_pixel(tl_gx, tl_gy);
        let (br_sx, br_sy) = src_gt.geo_to_pixel(br_gx, br_gy);
        (tl_sx, tl_sy, br_sx, br_sy)
    };

    let src_x0 = (tl_sx.min(br_sx).floor() as i64 - 2).max(0) as u32;
    let src_y0 = (tl_sy.min(br_sy).floor() as i64 - 2).max(0) as u32;
    let src_x1 = ((tl_sx.max(br_sx).ceil() as u32 + 2).min(src_meta.width)).max(src_x0 + 1);
    let src_y1 = ((tl_sy.max(br_sy).ceil() as u32 + 2).min(src_meta.height)).max(src_y0 + 1);

    // If source pixels don't overlap target at all, skip this scene
    if src_x0 >= src_meta.width || src_y0 >= src_meta.height {
        anyhow::bail!("Scene does not overlap target grid");
    }

    let src_bbox = PixelBbox {
        x0: src_x0,
        y0: src_y0,
        x1: src_x1,
        y1: src_y1,
    };

    // Download all bands + SCL concurrently
    let mut band_futures = Vec::new();
    let src_epsg_for_bands = src_meta.epsg;
    for bname in SPECTRAL_BANDS.iter().chain(std::iter::once(&"SCL")) {
        let url = band_urls
            .get(*bname)
            .with_context(|| format!("Missing band {bname} in signed URLs"))?
            .clone();
        let client = client.clone();
        let src_bbox_copy = src_bbox;

        band_futures.push(async move {
            // Read this band's metadata (may differ from B02 for 20m bands)
            let band_meta = cog::read_cog_meta(&client, &url).await?;
            let band_gt = GeoTransform::from_cog(&band_meta.pixel_scale, &band_meta.tiepoint);

            // Calculate bbox in this band's pixel coordinates
            let (bl_gx, bl_gy) =
                src_gt.pixel_to_geo(src_bbox_copy.x0 as f64, src_bbox_copy.y0 as f64);
            let (br2_gx, br2_gy) =
                src_gt.pixel_to_geo(src_bbox_copy.x1 as f64, src_bbox_copy.y1 as f64);

            let (b_x0, b_y0) = band_gt.geo_to_pixel(bl_gx, bl_gy);
            let (b_x1, b_y1) = band_gt.geo_to_pixel(br2_gx, br2_gy);

            // ±3 padding for 20m bands to give bilinear interpolation enough margin
            let bx0 = (b_x0.min(b_x1).floor() as i64 - 3).max(0) as u32;
            let by0 = (b_y0.min(b_y1).floor() as i64 - 3).max(0) as u32;
            let bx1 = ((b_x0.max(b_x1).ceil() as u32 + 3).min(band_meta.width)).max(bx0 + 1);
            let by1 = ((b_y0.max(b_y1).ceil() as u32 + 3).min(band_meta.height)).max(by0 + 1);

            let band_bbox = PixelBbox {
                x0: bx0,
                y0: by0,
                x1: bx1,
                y1: by1,
            };

            // Download the tiles for this region
            let raw_pixels = cog::read_cog_region(&client, &url, &band_meta, band_bbox).await?;
            let raw_w = (bx1 - bx0) as usize;
            let raw_h = (by1 - by0) as usize;

            // If band has different resolution (20m vs 10m), resample to 10m grid
            let (raw_crop_gx, raw_crop_gy) = band_gt.pixel_to_geo(bx0 as f64, by0 as f64);
            let raw_gt = GeoTransform {
                origin_x: raw_crop_gx,
                origin_y: raw_crop_gy,
                pixel_size_x: band_gt.pixel_size_x,
                pixel_size_y: band_gt.pixel_size_y,
            };

            Ok::<_, anyhow::Error>((raw_pixels, raw_w, raw_h, raw_gt))
        });
    }

    let band_results = futures::future::join_all(band_futures).await;

    // Collect all results, bail on first error
    let mut band_data: Vec<(Vec<f32>, usize, usize, GeoTransform)> = Vec::new();
    for (i, result) in band_results.into_iter().enumerate() {
        let label = if i < n_bands {
            SPECTRAL_BANDS[i]
        } else {
            "SCL"
        };
        let data = result.with_context(|| format!("Band {label} download failed"))?;
        band_data.push(data);
    }

    // Process spectral bands: resample each to target grid, freeing raw data immediately
    let n_pixels = dst_w * dst_h;
    let mut bands = Vec::with_capacity(n_bands);

    // Extract SCL (last element) first so we can consume spectral bands freely
    let scl_entry = band_data.pop().unwrap(); // SCL is always last

    // Consume spectral bands — each raw buffer is freed right after resampling
    for (raw_pixels, raw_w, raw_h, raw_gt) in band_data.into_iter() {
        let resampled = if epsg_mismatch {
            reproject::resample_bilinear_cross_crs(
                &raw_pixels, raw_w, raw_h, &raw_gt, src_epsg_for_bands,
                dst_w, dst_h, dst_gt, dst_epsg,
            )
        } else {
            reproject::resample_bilinear_par(
                &raw_pixels, raw_w, raw_h, &raw_gt, dst_w, dst_h, dst_gt,
            )
        };
        // raw_pixels is dropped here — frees source tile memory immediately
        bands.push(resampled);
    }

    // ── Detect out-of-footprint pixels ──────────────────────────────────
    // Sentinel-2 COGs write literal 0 for pixels outside the satellite's
    // ground footprint.  After bilinear resampling, footprint-edge pixels
    // get interpolated to near-zero across ALL bands.  If we keep them,
    // the Q1 composite picks these zeros over real data from overlapping
    // scenes, producing the "stripe" artifacts at orbit swath boundaries.
    //
    // Heuristic: if a pixel has *all 10 spectral bands* ≤ 1.0 (essentially
    // zero after BOA correction), it is outside the footprint.  Even the
    // darkest real surface (deep water) has ≥ 5–10 in at least one band
    // (B02 blue).  Mark these as NaN so the compositor ignores them.
    let footprint_threshold = 1.0f32;
    let mut n_outside_footprint = 0usize;
    for px in 0..n_pixels {
        let all_near_zero = bands.iter().all(|b| {
            let v = b[px];
            !v.is_finite() || v.abs() <= footprint_threshold
        });
        if all_near_zero {
            for band in bands.iter_mut() {
                band[px] = f32::NAN;
            }
            n_outside_footprint += 1;
        }
    }
    if n_outside_footprint > 0 {
        let pct = 100.0 * n_outside_footprint as f64 / n_pixels as f64;
        eprintln!(
            "      Footprint mask: {n_outside_footprint}/{n_pixels} pixels outside scene ({pct:.1}%)"
        );
    }

    // Process SCL band — nearest-neighbor (categorical mask)
    let scl_resampled = {
        let (ref raw_pixels, raw_w, raw_h, ref raw_gt) = scl_entry;
        if epsg_mismatch {
            reproject::resample_nearest_cross_crs(
                raw_pixels, raw_w, raw_h, raw_gt, src_epsg_for_bands,
                dst_w, dst_h, dst_gt, dst_epsg,
            )
        } else {
            reproject::resample_nearest_par(raw_pixels, raw_w, raw_h, raw_gt, dst_w, dst_h, dst_gt)
        }
    };
    drop(scl_entry); // free SCL source data

    // Build cloud mask from SCL
    let mut valid_mask = vec![true; n_pixels];
    for px in 0..n_pixels {
        let scl_val = scl_resampled[px].round() as u8;
        if SCL_EXCLUDE.contains(&scl_val) || !scl_resampled[px].is_finite() {
            valid_mask[px] = false;
        }
        // Also mark footprint-masked pixels as invalid
        if !bands[0][px].is_finite() {
            valid_mask[px] = false;
        }
    }

    // Apply mask to spectral bands (set masked pixels to NaN)
    for band in bands.iter_mut() {
        for px in 0..n_pixels {
            if !valid_mask[px] || !band[px].is_finite() {
                band[px] = f32::NAN;
            }
        }
    }

    Ok(SceneData { bands, valid_mask })
}

/// Write the composite as a pixel-interleaved float32 GeoTIFF.
///
/// Output layout matches composite.py's output: [H × W × (10 spectral + 1 valid_fraction)]
/// stored as pixel-interleaved (so tif_reader.rs can deinterleave it).
fn write_composite_tif(
    path: &Path,
    anchor: &AnchorRef,
    composite: &[f32],      // [n_bands * n_pixels], band-sequential
    valid_fraction: &[f32], // [n_pixels]
) -> Result<()> {
    use std::io::BufWriter;

    let w = anchor.width as u32;
    let h = anchor.height as u32;
    let n_bands: u16 = 11; // 10 spectral + valid_fraction
    let n_pixels = (w * h) as usize;

    // Build pixel-interleaved data
    let mut interleaved = vec![0u8; n_pixels * n_bands as usize * 4]; // float32 = 4 bytes
    for px in 0..n_pixels {
        for bi in 0..10 {
            let val = composite[bi * n_pixels + px];
            let bytes = val.to_le_bytes();
            let off = (px * n_bands as usize + bi) * 4;
            interleaved[off..off + 4].copy_from_slice(&bytes);
        }
        // Valid fraction in band 10
        let vf_bytes = valid_fraction[px].to_le_bytes();
        let off = (px * n_bands as usize + 10) * 4;
        interleaved[off..off + 4].copy_from_slice(&vf_bytes);
    }

    // Build a minimal classic TIFF with GeoTIFF tags
    // This is a simplified TIFF writer — just enough for tif_reader.rs to decode.
    let file =
        std::fs::File::create(path).with_context(|| format!("Cannot create {}", path.display()))?;
    let mut bw = BufWriter::new(file);

    // We'll use a minimal manual TIFF writer for simplicity
    write_geotiff_manual(&mut bw, w, h, n_bands, &interleaved, anchor)?;

    Ok(())
}

/// Write a minimal GeoTIFF by hand with DEFLATE compression.
fn write_geotiff_manual(
    w: &mut impl std::io::Write,
    width: u32,
    height: u32,
    n_bands: u16,
    pixel_data: &[u8],
    anchor: &AnchorRef,
) -> Result<()> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write as _;

    // Compress pixel data with zlib-wrapped DEFLATE (TIFF Compression=8)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(pixel_data)?;
    let compressed = encoder.finish()?;
    let compressed_bytes = compressed.len();

    // TIFF header (classic, little-endian)
    let ifd_offset: u32 = 8;
    w.write_all(b"II")?; // little endian
    w.write_all(&42u16.to_le_bytes())?; // magic
    w.write_all(&ifd_offset.to_le_bytes())?;

    // Count IFD entries
    let n_entries: u16 = 14;
    w.write_all(&n_entries.to_le_bytes())?;

    // Calculate offsets: IFD entries end, then tag data arrays, then pixel data
    let ifd_end = 8 + 2 + n_entries as u32 * 12 + 4; // +4 for next IFD offset
    let mut extra_off = ifd_end;

    // BitsPerSample array (11 x u16)
    let bps_offset = extra_off;
    extra_off += n_bands as u32 * 2;

    // SampleFormat array (11 x u16)
    let sf_offset = extra_off;
    extra_off += n_bands as u32 * 2;

    // ModelPixelScaleTag (3 x f64)
    let mps_offset = extra_off;
    extra_off += 24;

    // ModelTiepointTag (6 x f64)
    let mtp_offset = extra_off;
    extra_off += 48;

    // GeoKeyDirectory (4 x u16 header + 1 key * 4 u16)
    let gkd_offset = extra_off;
    extra_off += 16; // 4 header + 1 key = 8 u16 = 16 bytes

    // Pixel data
    let strip_offset = extra_off;

    // == Write IFD entries ==
    let write_entry =
        |w: &mut dyn std::io::Write, tag: u16, typ: u16, count: u32, value: u32| -> Result<()> {
            w.write_all(&tag.to_le_bytes())?;
            w.write_all(&typ.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
            w.write_all(&value.to_le_bytes())?;
            Ok(())
        };

    // ImageWidth
    write_entry(w, 256, 4, 1, width)?; // LONG
                                       // ImageLength
    write_entry(w, 257, 4, 1, height)?;
    // BitsPerSample (offset to array)
    write_entry(w, 258, 3, n_bands as u32, bps_offset)?;
    // Compression = DEFLATE (8)
    write_entry(w, 259, 3, 1, 8)?;
    // PhotometricInterpretation = 1 (min-is-black)
    write_entry(w, 262, 3, 1, 1)?;
    // StripOffsets
    write_entry(w, 273, 4, 1, strip_offset)?; // LONG
                                              // SamplesPerPixel
    write_entry(w, 277, 3, 1, n_bands as u32)?;
    // RowsPerStrip = height (single strip)
    write_entry(w, 278, 4, 1, height)?; // LONG to support height > 65535
    // StripByteCounts = compressed size
    write_entry(w, 279, 4, 1, compressed_bytes as u32)?;
    // PlanarConfiguration = 1 (pixel-interleaved)
    write_entry(w, 284, 3, 1, 1)?;
    // SampleFormat (offset to array)
    write_entry(w, 339, 3, n_bands as u32, sf_offset)?;
    // ModelPixelScaleTag
    write_entry(w, 33550, 12, 3, mps_offset)?; // DOUBLE
                                               // ModelTiepointTag
    write_entry(w, 33922, 12, 6, mtp_offset)?;
    // GeoKeyDirectoryTag
    write_entry(w, 34735, 3, 8, gkd_offset)?; // SHORT

    // Next IFD offset = 0 (no more IFDs)
    w.write_all(&0u32.to_le_bytes())?;

    // == Write extra tag data ==

    // BitsPerSample: all 32
    for _ in 0..n_bands {
        w.write_all(&32u16.to_le_bytes())?;
    }
    // SampleFormat: all 3 (IEEEFP)
    for _ in 0..n_bands {
        w.write_all(&3u16.to_le_bytes())?;
    }
    // ModelPixelScaleTag
    w.write_all(&anchor.geo_transform.pixel_size_x.to_le_bytes())?;
    w.write_all(&anchor.geo_transform.pixel_size_y.to_le_bytes())?;
    w.write_all(&0.0f64.to_le_bytes())?;
    // ModelTiepointTag (pixel 0,0 -> geo origin)
    w.write_all(&0.0f64.to_le_bytes())?; // pixel X
    w.write_all(&0.0f64.to_le_bytes())?; // pixel Y
    w.write_all(&0.0f64.to_le_bytes())?; // pixel Z
    w.write_all(&anchor.geo_transform.origin_x.to_le_bytes())?; // geo X
    w.write_all(&anchor.geo_transform.origin_y.to_le_bytes())?; // geo Y
    w.write_all(&0.0f64.to_le_bytes())?; // geo Z
                                         // GeoKeyDirectory
    w.write_all(&1u16.to_le_bytes())?; // KeyDirectoryVersion
    w.write_all(&1u16.to_le_bytes())?; // KeyRevision
    w.write_all(&0u16.to_le_bytes())?; // MinorRevision
    w.write_all(&1u16.to_le_bytes())?; // NumberOfKeys
    w.write_all(&3072u16.to_le_bytes())?; // ProjectedCSTypeGeoKey
    w.write_all(&0u16.to_le_bytes())?; // TIFFTagLocation (value inline)
    w.write_all(&1u16.to_le_bytes())?; // Count
    w.write_all(&(anchor.epsg as u16).to_le_bytes())?; // Value

    // == Write compressed pixel data ==
    w.write_all(&compressed)?;

    w.flush()?;
    Ok(())
}
