//! Cloud-Optimized GeoTIFF reader with HTTP range-based tile access.
//!
//! Reads COG metadata from partial HTTP downloads, then fetches
//! only the tiles needed for a given pixel bounding box.

use anyhow::{Context, Result};
use reqwest::Client;
use std::io::Read;

// ── How much of the COG header to download for IFD + tag data ──
const HEADER_BYTES: usize = 512 * 1024; // 512 KB covers all IFD + tile-offset arrays

// ── TIFF tag IDs we care about ──
const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_PREDICTOR: u16 = 317;
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;

// TIFF field types
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;
const TYPE_DOUBLE: u16 = 12;
const TYPE_LONG8: u16 = 16;

// ── Public types ──

/// Parsed metadata from a COG's IFD.
#[derive(Debug, Clone)]
pub struct CogMeta {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_offsets: Vec<u64>,
    pub tile_byte_counts: Vec<u64>,
    pub compression: u16,     // 8 = DEFLATE, 1 = none
    pub bits_per_sample: u16, // 8, 16, or 32
    pub sample_format: u16,   // 1 = uint, 3 = float
    pub predictor: u16,       // 1 = none, 2 = horizontal diff
    pub samples_per_pixel: u16,
    pub pixel_scale: [f64; 3], // from ModelPixelScaleTag
    pub tiepoint: [f64; 6],    // from ModelTiepointTag
    pub epsg: u32,             // from GeoKeyDirectory
    pub le: bool,              // true = little-endian, false = big-endian
}

impl CogMeta {
    /// Number of tiles in X and Y.
    pub fn tiles_across(&self) -> (u32, u32) {
        let nx = (self.width + self.tile_width - 1) / self.tile_width;
        let ny = (self.height + self.tile_height - 1) / self.tile_height;
        (nx, ny)
    }

    /// Flat tile index from (column, row).
    pub fn tile_index(&self, tx: u32, ty: u32) -> usize {
        let (nx, _) = self.tiles_across();
        (ty * nx + tx) as usize
    }
}

/// A pixel bounding box within a raster.
#[derive(Debug, Clone, Copy)]
pub struct PixelBbox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32, // exclusive
    pub y1: u32, // exclusive
}

// ── IFD parsing ──

/// Download the first HEADER_BYTES of a COG and parse its IFD.
pub async fn read_cog_meta(client: &Client, url: &str) -> Result<CogMeta> {
    // Download header
    let header = download_range(client, url, 0, HEADER_BYTES)
        .await
        .context("Failed to download COG header")?;

    parse_ifd(&header)
}

/// Parse TIFF IFD from a byte buffer (supports classic TIFF and BigTIFF).
fn parse_ifd(buf: &[u8]) -> Result<CogMeta> {
    if buf.len() < 8 {
        anyhow::bail!("Buffer too small for TIFF header");
    }

    // Byte order
    let le = match (buf[0], buf[1]) {
        (b'I', b'I') => true,
        (b'M', b'M') => false,
        _ => anyhow::bail!("Invalid TIFF byte order marker"),
    };

    let ru16 = |off: usize| -> u16 {
        if le {
            u16::from_le_bytes([buf[off], buf[off + 1]])
        } else {
            u16::from_be_bytes([buf[off], buf[off + 1]])
        }
    };
    let ru32 = |off: usize| -> u32 {
        if le {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        } else {
            u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        }
    };
    let ru64 = |off: usize| -> u64 {
        let b: [u8; 8] = buf[off..off + 8].try_into().unwrap();
        if le {
            u64::from_le_bytes(b)
        } else {
            u64::from_be_bytes(b)
        }
    };
    let rf64 = |off: usize| -> f64 {
        let b: [u8; 8] = buf[off..off + 8].try_into().unwrap();
        if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        }
    };

    let magic = ru16(2);
    let (ifd_offset, is_bigtiff) = if magic == 42 {
        (ru32(4) as usize, false)
    } else if magic == 43 {
        // BigTIFF: bytes 4-5 = offset size (8), bytes 8-15 = IFD offset
        (ru64(8) as usize, true)
    } else {
        anyhow::bail!("Unknown TIFF magic: {magic}");
    };

    // Parse IFD entries
    let n_entries = if is_bigtiff {
        ru64(ifd_offset) as usize
    } else {
        ru16(ifd_offset) as usize
    };

    let entry_start = if is_bigtiff {
        ifd_offset + 8
    } else {
        ifd_offset + 2
    };
    let entry_size = if is_bigtiff { 20 } else { 12 };

    // Collect raw IFD entries
    let mut meta = CogMeta {
        width: 0,
        height: 0,
        tile_width: 256,
        tile_height: 256,
        tile_offsets: Vec::new(),
        tile_byte_counts: Vec::new(),
        compression: 1,
        bits_per_sample: 16,
        sample_format: 1,
        predictor: 1,
        samples_per_pixel: 1,
        pixel_scale: [0.0; 3],
        tiepoint: [0.0; 6],
        epsg: 0,
        le,
    };

    for i in 0..n_entries {
        let eoff = entry_start + i * entry_size;
        if eoff + entry_size > buf.len() {
            break;
        }

        let tag = ru16(eoff);
        let field_type = ru16(eoff + 2);
        let count = if is_bigtiff {
            ru64(eoff + 4) as u32
        } else {
            ru32(eoff + 4)
        };
        let value_offset = if is_bigtiff { eoff + 12 } else { eoff + 8 };

        // Helper: read a single u32/u16 value (inline or from offset)
        let read_u32_val = || -> u32 {
            match field_type {
                TYPE_SHORT => ru16(value_offset) as u32,
                TYPE_LONG => ru32(value_offset),
                _ => ru32(value_offset),
            }
        };

        // Helper: resolve the offset where array data lives
        let data_off = || -> usize {
            let val_size = type_size(field_type) * count as usize;
            let inline_limit = if is_bigtiff { 8 } else { 4 };
            if val_size <= inline_limit {
                value_offset
            } else if is_bigtiff {
                ru64(value_offset) as usize
            } else {
                ru32(value_offset) as usize
            }
        };

        match tag {
            TAG_IMAGE_WIDTH => meta.width = read_u32_val(),
            TAG_IMAGE_LENGTH => meta.height = read_u32_val(),
            TAG_BITS_PER_SAMPLE => meta.bits_per_sample = ru16(value_offset),
            TAG_COMPRESSION => meta.compression = ru16(value_offset),
            TAG_SAMPLES_PER_PIXEL => meta.samples_per_pixel = ru16(value_offset),
            TAG_TILE_WIDTH => meta.tile_width = read_u32_val(),
            TAG_TILE_LENGTH => meta.tile_height = read_u32_val(),
            TAG_SAMPLE_FORMAT => meta.sample_format = ru16(value_offset),
            TAG_PREDICTOR => meta.predictor = ru16(value_offset),

            TAG_TILE_OFFSETS => {
                let off = data_off();
                meta.tile_offsets = read_u64_array(buf, off, count as usize, field_type, le);
            }
            TAG_TILE_BYTE_COUNTS => {
                let off = data_off();
                meta.tile_byte_counts = read_u64_array(buf, off, count as usize, field_type, le);
            }
            TAG_MODEL_PIXEL_SCALE => {
                let off = data_off();
                if off + 24 <= buf.len() {
                    for j in 0..3 {
                        meta.pixel_scale[j] = rf64(off + j * 8);
                    }
                }
            }
            TAG_MODEL_TIEPOINT => {
                let off = data_off();
                if off + 48 <= buf.len() {
                    for j in 0..6 {
                        meta.tiepoint[j] = rf64(off + j * 8);
                    }
                }
            }
            TAG_GEO_KEY_DIRECTORY => {
                let off = data_off();
                // GeoKeyDirectory: array of u16 [KeyDirectoryVersion, Revision, MinorRevision, NumberOfKeys, ...]
                // Key 2048 = GeographicTypeGeoKey, Key 3072 = ProjectedCSTypeGeoKey
                let n_keys = if off + 6 < buf.len() {
                    ru16(off + 6) as usize
                } else {
                    0
                };
                for k in 0..n_keys {
                    let koff = off + 8 + k * 8;
                    if koff + 8 > buf.len() {
                        break;
                    }
                    let key_id = ru16(koff);
                    let _tiff_tag_location = ru16(koff + 2);
                    let _count = ru16(koff + 4);
                    let value = ru16(koff + 6);
                    if key_id == 3072 && value > 0 {
                        // ProjectedCSTypeGeoKey
                        meta.epsg = value as u32;
                    } else if key_id == 2048 && meta.epsg == 0 && value > 0 {
                        // GeographicTypeGeoKey (fallback)
                        meta.epsg = value as u32;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(meta)
}

/// Size in bytes for a TIFF field type.
fn type_size(ft: u16) -> usize {
    match ft {
        1 | 2 | 6 | 7 => 1,      // BYTE, ASCII, SBYTE, UNDEFINED
        TYPE_SHORT | 8 => 2,     // SHORT, SSHORT
        TYPE_LONG | 9 => 4,      // LONG, SLONG
        TYPE_RATIONAL | 10 => 8, // RATIONAL, SRATIONAL
        11 => 4,                 // FLOAT
        TYPE_DOUBLE => 8,        // DOUBLE
        TYPE_LONG8 | 17 => 8,    // LONG8, SLONG8
        _ => 1,
    }
}

/// Read an array of u64 values from the buffer (handles SHORT, LONG, LONG8).
fn read_u64_array(buf: &[u8], off: usize, count: usize, field_type: u16, le: bool) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let elem_size = type_size(field_type);
    for i in 0..count {
        let p = off + i * elem_size;
        if p + elem_size > buf.len() {
            break;
        }
        let val = match field_type {
            TYPE_SHORT => {
                if le {
                    u16::from_le_bytes([buf[p], buf[p + 1]]) as u64
                } else {
                    u16::from_be_bytes([buf[p], buf[p + 1]]) as u64
                }
            }
            TYPE_LONG => {
                if le {
                    u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as u64
                } else {
                    u32::from_be_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as u64
                }
            }
            TYPE_LONG8 => {
                let b: [u8; 8] = buf[p..p + 8].try_into().unwrap();
                if le {
                    u64::from_le_bytes(b)
                } else {
                    u64::from_be_bytes(b)
                }
            }
            _ => {
                if le {
                    u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as u64
                } else {
                    u32::from_be_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as u64
                }
            }
        };
        out.push(val);
    }
    out
}

// ── Tile access ──

/// Determine which tile indices overlap a pixel bounding box.
pub fn tiles_for_pixel_bbox(meta: &CogMeta, bbox: PixelBbox) -> Vec<(u32, u32)> {
    let tx0 = bbox.x0 / meta.tile_width;
    let ty0 = bbox.y0 / meta.tile_height;
    let tx1 = (bbox.x1.saturating_sub(1)) / meta.tile_width;
    let ty1 = (bbox.y1.saturating_sub(1)) / meta.tile_height;
    let mut tiles = Vec::new();
    for ty in ty0..=ty1 {
        for tx in tx0..=tx1 {
            tiles.push((tx, ty));
        }
    }
    tiles
}

/// Download tiles and assemble into a pixel buffer for the requested bbox.
///
/// Returns a flat f32 buffer of size (bbox.height × bbox.width).
pub async fn read_cog_region(
    client: &Client,
    url: &str,
    meta: &CogMeta,
    bbox: PixelBbox,
) -> Result<Vec<f32>> {
    let out_w = (bbox.x1 - bbox.x0) as usize;
    let out_h = (bbox.y1 - bbox.y0) as usize;
    let mut output = vec![f32::NAN; out_h * out_w];

    let tiles = tiles_for_pixel_bbox(meta, bbox);
    if tiles.is_empty() {
        return Ok(output);
    }

    // Download all needed tiles concurrently
    let tile_futures: Vec<_> = tiles
        .iter()
        .map(|&(tx, ty)| {
            let client = client.clone();
            let url = url.to_string();
            let idx = meta.tile_index(tx, ty);
            let in_bounds = idx < meta.tile_offsets.len() && idx < meta.tile_byte_counts.len();
            let offset = if in_bounds { meta.tile_offsets[idx] } else { 0 };
            let size = if in_bounds { meta.tile_byte_counts[idx] as usize } else { 0 };
            let compression = meta.compression;
            let bits = meta.bits_per_sample;
            let sample_fmt = meta.sample_format;
            let predictor = meta.predictor;
            let is_le = meta.le;
            let tw = meta.tile_width as usize;
            let th = meta.tile_height as usize;
            async move {
                if !in_bounds {
                    // Out-of-bounds tile index — return NaN-filled tile
                    return Ok::<_, anyhow::Error>((tx, ty, vec![f32::NAN; tw * th]));
                }
                let raw = download_range(&client, &url, offset as usize, size).await?;
                let pixels = decode_tile(&raw, compression, bits, sample_fmt, predictor, tw, th, is_le)?;
                Ok::<_, anyhow::Error>((tx, ty, pixels))
            }
        })
        .collect();

    let results = futures::future::join_all(tile_futures).await;

    // Assemble tiles into output buffer
    for result in results {
        let (tx, ty, pixels) = result?;
        let tile_px_x = tx * meta.tile_width;
        let tile_px_y = ty * meta.tile_height;
        let tw = meta.tile_width as usize;
        let th = meta.tile_height as usize;

        for dy in 0..th {
            let src_y = tile_px_y as usize + dy;
            if src_y < bbox.y0 as usize || src_y >= bbox.y1 as usize {
                continue;
            }
            let dst_y = src_y - bbox.y0 as usize;

            for dx in 0..tw {
                let src_x = tile_px_x as usize + dx;
                if src_x < bbox.x0 as usize || src_x >= bbox.x1 as usize {
                    continue;
                }
                let dst_x = src_x - bbox.x0 as usize;

                let val = pixels[dy * tw + dx];
                output[dst_y * out_w + dst_x] = val;
            }
        }
    }

    Ok(output)
}

// ── Tile decoding ──

/// Unpack a contiguous stream of 15-bit tightly packed integers into a Vec<u16>.
/// TIFF spec uses MSB-first packing within bytes, but since bits are just sequential
/// we read 3 bytes at a time (at most) and mask the desired 15 bits.
fn unpack_15bit_tight(raw: &[u8], n_pixels: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n_pixels);
    let mut bit_buf = 0u32;
    let mut bits_in_buf = 0usize;
    let mut byte_idx = 0usize;

    for _ in 0..n_pixels {
        // Accumulate bytes into buffer until we have at least 15 bits
        while bits_in_buf < 15 {
            if byte_idx < raw.len() {
                bit_buf = (bit_buf << 8) | (raw[byte_idx] as u32);
                byte_idx += 1;
            } else {
                // Pad with zeros if we run out of input bytes
                bit_buf <<= 8;
            }
            bits_in_buf += 8;
        }

        // Extract the top 15 bits from our buffer
        let shift = bits_in_buf - 15;
        let sample = (bit_buf >> shift) & 0x7FFF;
        out.push(sample as u16);

        // Remove the consumed bits
        bits_in_buf -= 15;
        bit_buf &= (1 << bits_in_buf) - 1;
    }
    out
}

/// Decode a compressed tile into f32 pixels.
fn decode_tile(
    raw: &[u8],
    compression: u16,
    bits_per_sample: u16,
    sample_format: u16,
    predictor: u16,
    tile_width: usize,
    tile_height: usize,
    le: bool,
) -> Result<Vec<f32>> {
    // Decompress
    let decompressed = match compression {
        1 => raw.to_vec(), // no compression
        8 | 32946 => {
            // DEFLATE — try zlib wrapper first, then raw deflate
            let mut buf = Vec::new();
            let ok = {
                use flate2::read::ZlibDecoder;
                let mut dec = ZlibDecoder::new(raw);
                dec.read_to_end(&mut buf).is_ok()
            };
            if !ok {
                buf.clear();
                use flate2::read::DeflateDecoder;
                let mut dec = DeflateDecoder::new(raw);
                dec.read_to_end(&mut buf)
                    .context("DEFLATE decompression failed")?;
            }
            buf
        }
        50000 => {
            // ZSTD compression
            // Used by Sentinel-1 GRD/RTC from late 2023+ from Planetary Computer
            let mut dec = zstd::stream::Decoder::new(raw).context("Failed to init ZSTD decoder")?;
            let mut buf = Vec::with_capacity(tile_width * tile_height * 2);
            dec.read_to_end(&mut buf)
                .context("ZSTD decompression failed")?;
            buf
        }
        _ => anyhow::bail!("Unsupported TIFF compression: {compression}"),
    };

    let n_pixels = tile_width * tile_height;

    // Apply horizontal differencing predictor (undo)
    // TIFF predictor=2 stores pixel[x] = pixel[x] - pixel[x-1] (deltas).
    // Undoing means cumulative sum across each row at the *sample* level.
    let mut bytes = decompressed.clone();
    if predictor == 2 {
        let bps = bits_per_sample as usize; // bits per sample
        let bytes_per_sample = bps / 8;
        match bytes_per_sample {
            1 => {
                // uint8: byte-level cumsum
                let row_bytes = tile_width;
                for row in 0..tile_height {
                    let rs = row * row_bytes;
                    for x in 1..row_bytes {
                        let idx = rs + x;
                        if idx < bytes.len() {
                            bytes[idx] = bytes[idx].wrapping_add(bytes[idx - 1]);
                        }
                    }
                }
            }
            2 => {
                // uint16: sample-level cumsum (operate on u16 values)
                let samples_per_row = tile_width;
                for row in 0..tile_height {
                    let rs = row * samples_per_row * 2; // byte offset of row start
                    for x in 1..samples_per_row {
                        let cur = rs + x * 2;
                        let prev = rs + (x - 1) * 2;
                        if cur + 1 < bytes.len() {
                            let cur_val = if le {
                                u16::from_le_bytes([bytes[cur], bytes[cur + 1]])
                            } else {
                                u16::from_be_bytes([bytes[cur], bytes[cur + 1]])
                            };
                            let prev_val = if le {
                                u16::from_le_bytes([bytes[prev], bytes[prev + 1]])
                            } else {
                                u16::from_be_bytes([bytes[prev], bytes[prev + 1]])
                            };
                            let result = cur_val.wrapping_add(prev_val);
                            let rb = if le { result.to_le_bytes() } else { result.to_be_bytes() };
                            bytes[cur] = rb[0];
                            bytes[cur + 1] = rb[1];
                        }
                    }
                }
            }
            4 => {
                // float32: sample-level cumsum (operate on f32 values)
                let samples_per_row = tile_width;
                for row in 0..tile_height {
                    let rs = row * samples_per_row * 4;
                    for x in 1..samples_per_row {
                        let cur = rs + x * 4;
                        let prev = rs + (x - 1) * 4;
                        if cur + 3 < bytes.len() {
                            let cur_val = if le {
                                f32::from_le_bytes([
                                    bytes[cur],
                                    bytes[cur + 1],
                                    bytes[cur + 2],
                                    bytes[cur + 3],
                                ])
                            } else {
                                f32::from_be_bytes([
                                    bytes[cur],
                                    bytes[cur + 1],
                                    bytes[cur + 2],
                                    bytes[cur + 3],
                                ])
                            };
                            let prev_val = if le {
                                f32::from_le_bytes([
                                    bytes[prev],
                                    bytes[prev + 1],
                                    bytes[prev + 2],
                                    bytes[prev + 3],
                                ])
                            } else {
                                f32::from_be_bytes([
                                    bytes[prev],
                                    bytes[prev + 1],
                                    bytes[prev + 2],
                                    bytes[prev + 3],
                                ])
                            };
                            let result = cur_val + prev_val;
                            let rb = if le { result.to_le_bytes() } else { result.to_be_bytes() };
                            bytes[cur..cur + 4].copy_from_slice(&rb);
                        }
                    }
                }
            }
            // Add custom handler for 15-bit tight packing (stored as 15bps but 1 byte_per_sample here is misleading,
            // the bytes_per_sample logic above fails). We must intercept *before* the predictor.
            _ => {
                // If it's 15bps, the generic fallback is wrong.
                if bits_per_sample == 15 {
                    // Handled down below, we must unpack first, then run predictor!
                } else {
                    // fallback: byte-level (may be incorrect for some formats)
                    let row_bytes = tile_width * bytes_per_sample;
                    for row in 0..tile_height {
                        let rs = row * row_bytes;
                        for x in bytes_per_sample..row_bytes {
                            let idx = rs + x;
                            if idx < bytes.len() {
                                bytes[idx] = bytes[idx].wrapping_add(bytes[idx - bytes_per_sample]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert to f32
    let pixels = match (bits_per_sample, sample_format) {
        (8, 1) => {
            // uint8
            bytes.iter().take(n_pixels).map(|&v| v as f32).collect()
        }
        (15, 1) => {
            // 15bps tight packing (Newer ESA baseline).
            // We unpacked raw tight bits OR they are byte-aligned. Let's unpack first.
            let mut unpacked = unpack_15bit_tight(&decompressed, n_pixels);

            // Re-apply predictor=2 correctly on the *unpacked* 16-bit values
            if predictor == 2 {
                let samples_per_row = tile_width;
                for row in 0..tile_height {
                    let rs = row * samples_per_row;
                    for x in 1..samples_per_row {
                        unpacked[rs + x] = unpacked[rs + x].wrapping_add(unpacked[rs + x - 1]);
                    }
                }
            }

            unpacked.into_iter().map(|v| v as f32).collect()
        }
        (16, 1) => {
            // uint16 LE
            // If predictor was 2, bytes array is already cumulative sum
            let mut out = Vec::with_capacity(n_pixels);
            for i in 0..n_pixels {
                let off = i * 2;
                if off + 1 < bytes.len() {
                    let v = if le {
                        u16::from_le_bytes([bytes[off], bytes[off + 1]])
                    } else {
                        u16::from_be_bytes([bytes[off], bytes[off + 1]])
                    };
                    out.push(v as f32);
                } else {
                    out.push(f32::NAN);
                }
            }
            out
        }
        (32, 3) => {
            // float32 LE
            let mut out = Vec::with_capacity(n_pixels);
            for i in 0..n_pixels {
                let off = i * 4;
                if off + 3 < bytes.len() {
                    let v = if le {
                        f32::from_le_bytes([
                            bytes[off],
                            bytes[off + 1],
                            bytes[off + 2],
                            bytes[off + 3],
                        ])
                    } else {
                        f32::from_be_bytes([
                            bytes[off],
                            bytes[off + 1],
                            bytes[off + 2],
                            bytes[off + 3],
                        ])
                    };
                    out.push(v);
                } else {
                    out.push(f32::NAN);
                }
            }
            out
        }
        _ => anyhow::bail!("Unsupported pixel format: {bits_per_sample}bps, fmt={sample_format}"),
    };

    Ok(pixels)
}

// ── HTTP helpers ──

/// Download a byte range from a URL (with retry on transient errors).
pub async fn download_range(
    client: &Client,
    url: &str,
    offset: usize,
    length: usize,
) -> Result<Vec<u8>> {
    let end = offset + length - 1;
    let max_retries = 3u32;

    for attempt in 0..=max_retries {
        let result = client
            .get(url)
            .header("Range", format!("bytes={offset}-{end}"))
            .send()
            .await;

        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() || status.as_u16() == 206 {
                    let data = resp.bytes().await.context("Failed to read response body")?;
                    return Ok(data.to_vec());
                }
                // Retry on server errors and rate limits
                if (status.as_u16() == 429 || status.as_u16() >= 500) && attempt < max_retries {
                    let wait = 1u64 << attempt; // 1s, 2s, 4s
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    continue;
                }
                anyhow::bail!("HTTP range request returned {status}");
            }
            Err(e) => {
                if attempt < max_retries {
                    let wait = 1u64 << attempt;
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    continue;
                }
                return Err(e).context("HTTP range request failed after retries");
            }
        }
    }

    anyhow::bail!("download_range exhausted retries for {url}")
}

/// Read CogMeta from a local GeoTIFF file (for anchor references).
pub fn read_local_tif_meta(path: &std::path::Path) -> Result<CogMeta> {
    let data =
        std::fs::read(path).with_context(|| format!("Cannot read TIF: {}", path.display()))?;

    // Only need the first HEADER_BYTES for IFD parsing
    let header = if data.len() > HEADER_BYTES {
        &data[..HEADER_BYTES]
    } else {
        &data
    };
    parse_ifd(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_15bit_tight() {
        let _v1 = 0x5555u32;
        let _v2 = 0x2AAAu32; // This doesn't matter for the new test, let's write exact bytes

        // Stream: [ 10101010 ] [ 10101010 ] [ 10101010 ] ...
        // First 15 bits: 101010101010101 -> 0x5555
        // This requires bit stream MSB-first:
        // byte 0: 10101010 = 0xAA
        // byte 1: 10101010 = 0xAA
        let bytes = [0xAA, 0xAA, 0xAA, 0xAA];
        let unpacked = unpack_15bit_tight(&bytes, 2);
        assert_eq!(unpacked[0], 0x5555);
        // leftover bit from first 15 bits is 0.
        // next 14 bits are 10101010101010.
        // so sample 2 is 010101010101010 = 0x2AAA.
        assert_eq!(unpacked[1], 0x2AAA);
    }
}
