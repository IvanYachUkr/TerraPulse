//! Native GeoTIFF reader using the `tiff` crate.
//! Reads multi-band pixel-interleaved float32 TIFFs without Python.

use anyhow::{Context, Result};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};

/// Read a GeoTIFF and return band-sequential float32 data.
///
/// Returns (n_bands, height, width, data) where data is [n_bands * H * W]
/// in band-sequential order: band0[pixel0..pixel_n], band1[pixel0..pixel_n], ...
pub fn read_tif_bands(path: &Path, max_bands: usize) -> Result<(usize, usize, usize, Vec<f32>)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open TIF: {}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("Cannot decode TIF: {}", path.display()))?;

    let (w, h) = decoder.dimensions()
        .with_context(|| format!("Cannot read dimensions: {}", path.display()))?;
    let w = w as usize;
    let h = h as usize;

    // Read all pixel data
    let image = decoder.read_image()
        .with_context(|| format!("Cannot read image data: {}", path.display()))?;

    let interleaved = match image {
        DecodingResult::F32(data) => data,
        _ => anyhow::bail!("Expected Float32 TIF, got non-F32 data type"),
    };

    let n_pixels = h * w;
    let total_samples = interleaved.len();
    let n_bands_total = total_samples / n_pixels;

    if total_samples != n_bands_total * n_pixels {
        anyhow::bail!(
            "TIF data size mismatch: {} samples, {}x{} pixels, cannot determine band count",
            total_samples, h, w
        );
    }

    let nb = n_bands_total.min(max_bands);

    // De-interleave: pixel-interleaved [px0_b0, px0_b1, ..., px1_b0, px1_b1, ...]
    // -> band-sequential [b0_px0, b0_px1, ..., b1_px0, b1_px1, ...]
    let mut band_seq = vec![0.0f32; nb * n_pixels];
    for px in 0..n_pixels {
        let src_base = px * n_bands_total;
        for b in 0..nb {
            band_seq[b * n_pixels + px] = interleaved[src_base + b];
        }
    }

    Ok((nb, h, w, band_seq))
}

/// Read only the valid_fraction band (band 11, 0-indexed = 10) from a TIF.
/// Returns None if the TIF has fewer than 11 bands.
pub fn read_valid_fraction(path: &Path) -> Result<Option<Vec<f32>>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open TIF: {}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("Cannot decode TIF: {}", path.display()))?;

    let (w, h) = decoder.dimensions()?;
    let w = w as usize;
    let h = h as usize;

    let image = decoder.read_image()?;
    let interleaved = match image {
        DecodingResult::F32(data) => data,
        _ => return Ok(None),
    };

    let n_pixels = h * w;
    let n_bands_total = interleaved.len() / n_pixels;

    if n_bands_total < 11 {
        return Ok(None);
    }

    // Extract band 10 (0-indexed, = VALID_FRACTION)
    let vf_band = 10;
    let mut vf = vec![0.0f32; n_pixels];
    for px in 0..n_pixels {
        let v = interleaved[px * n_bands_total + vf_band];
        // Replace nodata with NaN
        vf[px] = if v > -9000.0 { v } else { f32::NAN };
    }

    Ok(Some(vf))
}
