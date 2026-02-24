//! Native GeoTIFF reader using the `tiff` crate.
//! Reads multi-band pixel-interleaved float32 TIFFs without Python.

use anyhow::{Context, Result};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult, Limits};

/// Decode a GeoTIFF into raw pixel-interleaved float32 data.
fn decode_interleaved_f32(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open TIF: {}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("Cannot decode TIF: {}", path.display()))?;
    // Large anchors (e.g. 2550x2850 × 11 bands) exceed default decoder limits
    let mut limits = Limits::default();
    limits.decoding_buffer_size = 512 * 1024 * 1024; // 512 MB
    limits.intermediate_buffer_size = 512 * 1024 * 1024;
    decoder = decoder.with_limits(limits);

    let (w, h) = decoder
        .dimensions()
        .with_context(|| format!("Cannot read dimensions: {}", path.display()))?;

    let image = decoder
        .read_image()
        .with_context(|| format!("Cannot read image data: {}", path.display()))?;

    let interleaved = match image {
        DecodingResult::F32(data) => data,
        _ => anyhow::bail!("Expected Float32 TIF, got non-F32 data type"),
    };

    Ok((w as usize, h as usize, interleaved))
}

/// De-interleave pixel data from [px0_b0, px0_b1, ..., px1_b0, ...] to
/// band-sequential [b0_px0, b0_px1, ..., b1_px0, ...].
fn deinterleave(interleaved: &[f32], n_pixels: usize, n_bands_total: usize, nb: usize) -> Vec<f32> {
    let mut band_seq = vec![0.0f32; nb * n_pixels];
    for b in 0..nb {
        let dst = &mut band_seq[b * n_pixels..(b + 1) * n_pixels];
        for px in 0..n_pixels {
            dst[px] = interleaved[px * n_bands_total + b];
        }
    }
    band_seq
}

/// Decode once and extract both spectral bands AND valid_fraction,
/// avoiding decoding the same file twice.
pub fn read_tif_bands_and_valid_fraction(
    path: &Path,
    max_bands: usize,
) -> Result<(usize, usize, usize, Vec<f32>, Option<Vec<f32>>)> {
    let (w, h, interleaved) = decode_interleaved_f32(path)?;
    let n_pixels = h * w;
    let total_samples = interleaved.len();

    if n_pixels == 0 || total_samples % n_pixels != 0 {
        anyhow::bail!("Invalid TIFF layout for {}", path.display());
    }

    let n_bands_total = total_samples / n_pixels;
    let nb = n_bands_total.min(max_bands);
    let band_seq = deinterleave(&interleaved, n_pixels, n_bands_total, nb);

    let vf = if n_bands_total >= 11 {
        let vf_band = 10;
        let mut vf = vec![0.0f32; n_pixels];
        for px in 0..n_pixels {
            let v = interleaved[px * n_bands_total + vf_band];
            vf[px] = if v > -9000.0 { v } else { f32::NAN };
        }
        Some(vf)
    } else {
        None
    };

    Ok((nb, h, w, band_seq, vf))
}
