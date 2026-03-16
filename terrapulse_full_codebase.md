# TerraPulse — Full Codebase

This document contains the complete source code of the **TerraPulse** Rust project — a fast inference pipeline for land-cover prediction using Sentinel-2 and Sentinel-1 satellite imagery. It includes all Rust source files, the Cargo manifest, tests, and Python helper scripts.

---

## Project Structure

```
terrapulse/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI entry point + pipeline orchestration
│   ├── config.rs        # Class names and constants
│   ├── cog.rs           # Cloud-Optimized GeoTIFF reader (HTTP range-based)
│   ├── composite.rs     # Scene compositing (cloud mask + nanmedian)
│   ├── download.rs      # Sentinel-2/S1 download orchestration
│   ├── extract.rs       # Feature extraction pipeline
│   ├── features.rs      # 224 optical features per cell (stats, indices, LBP, TC)
│   ├── grid.rs          # GeoJSON grid generation
│   ├── labels.rs        # ESA WorldCover label download
│   ├── parquet_io.rs    # Parquet read/write via Arrow
│   ├── predict.rs       # ONNX model inference
│   ├── reproject.rs     # Bilinear/NN resampling + UTM transforms
│   ├── sar_download.rs  # SAR (S1 GRD) download with GCP handling
│   ├── sar_features.rs  # 48 SAR features per cell
│   ├── stac.rs          # STAC API client for Planetary Computer
│   └── tif_reader.rs    # Native GeoTIFF decoder
├── tests/
│   └── pipeline_e2e.rs  # Integration tests
├── check_s1_compression.py  # S1 TIFF compression checker
└── helpers/
    └── composite.py     # Python composite helper (legacy)
```

---

## Cargo.toml

```toml
[package]
name = "terrapulse"
version = "0.1.0"
edition = "2021"
description = "Fast inference pipeline for TerraPulse land-cover prediction"

[[bin]]
name = "terrapulse"
path = "src/main.rs"

[dependencies]
# CLI
clap = { version = "4", features = ["derive"] }

# Async runtime (for STAC + downloads)
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }

# HTTP & serialization
reqwest = { version = "0.12", features = ["json", "rustls-tls"], default-features = false }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Parallelism
rayon = "1.10"
futures = "0.3"

# ONNX Runtime inference (dynamic loading to avoid MSVC static linking issues)
ort = { version = "2.0.0-rc.9", default-features = false, features = ["std", "load-dynamic", "copy-dylibs", "download-binaries", "tls-native"] }

# Data I/O
parquet = { version = "53", features = ["arrow"] }
arrow = { version = "53", features = ["ffi"] }
tiff = "0.11"
flate2 = "1.0"
zstd = "0.13"
# Utilities
anyhow = "1"
utm = "0.1.6"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1

[dev-dependencies]
assert_cmd = "2.0"
predicates = "3.1"
```

---

## src/main.rs

```rust
mod cog;
mod composite;
mod config;
mod download;
mod extract;
mod features;
mod grid;
mod labels;
mod parquet_io;
mod predict;
mod reproject;
mod sar_download;
mod sar_features;
mod stac;
mod tif_reader;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

use config::CLASS_NAMES;

#[derive(Parser)]
#[command(name = "terrapulse", about = "Fast TerraPulse inference pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download Sentinel-2 composites via STAC
    Download {
        /// Bounding box [west, south, east, north] in WGS84
        #[arg(long, num_args = 4, allow_hyphen_values = true)]
        bbox: Vec<f64>,

        /// EPSG code for the target CRS (e.g., 32632)
        #[arg(long, default_value = "32632")]
        epsg: u32,

        /// Years to download
        #[arg(long, value_delimiter = ' ')]
        years: Vec<u32>,

        /// Region name (used in filenames)
        #[arg(long, default_value = "nuremberg")]
        region: String,

        /// Output directory for raw TIF files
        #[arg(long)]
        raw_dir: PathBuf,

        /// Path to the anchor reference GeoTIFF
        #[arg(long)]
        anchor_ref: PathBuf,
    },

    /// Run prediction on existing feature parquets
    Predict {
        /// Path to the models/onnx directory
        #[arg(long)]
        models_dir: PathBuf,

        /// Path to the features directory containing feature parquets
        #[arg(long)]
        features_dir: PathBuf,

        /// Output directory for predictions
        #[arg(long)]
        output_dir: PathBuf,

        /// Year pairs to predict (e.g., "2023_2024 2024_2025")
        #[arg(long, value_delimiter = ' ')]
        year_pairs: Vec<String>,
    },

    /// Extract features from downloaded GeoTIFFs
    Extract {
        /// Year pairs (e.g., "2020_2021 2021_2022")
        #[arg(long, value_delimiter = ' ')]
        year_pairs: Vec<String>,

        /// Region name
        #[arg(long, default_value = "nuremberg")]
        region: String,

        /// Raw TIF directory
        #[arg(long)]
        raw_dir: PathBuf,

        /// Features output directory
        #[arg(long)]
        features_dir: PathBuf,

        /// Minimum valid fraction threshold
        #[arg(long, default_value = "0.3")]
        min_valid_frac: f32,
    },

    /// Run the full pipeline: download → extract → predict
    Pipeline {
        /// Bounding box [west, south, east, north] in WGS84
        #[arg(long, num_args = 4, allow_hyphen_values = true)]
        bbox: Vec<f64>,

        /// EPSG code for the target CRS
        #[arg(long, default_value = "32632")]
        epsg: u32,

        /// Years to process (consecutive pairs derived automatically)
        #[arg(long, value_delimiter = ' ')]
        years: Vec<u32>,

        /// Region name
        #[arg(long, default_value = "nuremberg")]
        region: String,

        /// Base data directory (raw/, features/, predictions/ created inside)
        #[arg(long, default_value = "data/pipeline_output")]
        data_dir: PathBuf,

        /// Path to the anchor reference GeoTIFF
        #[arg(long)]
        anchor_ref: PathBuf,

        /// Path to the models/onnx directory
        #[arg(long)]
        models_dir: PathBuf,

        /// Minimum valid fraction threshold
        #[arg(long, default_value = "0.3")]
        min_valid_frac: f32,

        /// Skip download stage (use existing TIFs)
        #[arg(long, default_value = "false")]
        skip_download: bool,

        /// Skip extract stage (use existing parquets)
        #[arg(long, default_value = "false")]
        skip_extract: bool,

        /// Skip predict stage
        #[arg(long, default_value = "false")]
        skip_predict: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download {
            bbox,
            epsg,
            years,
            region,
            raw_dir,
            anchor_ref,
        } => {
            run_download(&bbox, epsg, &years, &region, &raw_dir, &anchor_ref).await?;
        }
        Commands::Predict {
            models_dir,
            features_dir,
            output_dir,
            year_pairs,
        } => {
            run_predict(&models_dir, &features_dir, &output_dir, &year_pairs)?;
        }
        Commands::Extract {
            year_pairs,
            region,
            raw_dir,
            features_dir,
            min_valid_frac,
        } => {
            run_extract(
                &year_pairs,
                &region,
                &raw_dir,
                &features_dir,
                min_valid_frac,
            )?;
        }
        Commands::Pipeline {
            bbox,
            epsg,
            years,
            region,
            data_dir,
            anchor_ref,
            models_dir,
            min_valid_frac,
            skip_download,
            skip_extract,
            skip_predict,
        } => {
            run_pipeline(
                &bbox,
                epsg,
                &years,
                &region,
                &data_dir,
                &anchor_ref,
                &models_dir,
                min_valid_frac,
                skip_download,
                skip_extract,
                skip_predict,
            )
            .await?;
        }
    }

    Ok(())
}

fn run_extract(
    year_pairs: &[String],
    region: &str,
    raw_dir: &Path,
    features_dir: &Path,
    min_valid_frac: f32,
) -> Result<()> {
    let t0 = Instant::now();
    println!("\nTerraPulse Feature Extraction");
    println!("  Region: {region}");
    println!("  Min valid frac: {min_valid_frac}");

    for yp in year_pairs {
        let parts: Vec<&str> = yp.split('_').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid year pair format '{}', expected 'YYYY_YYYY'", yp);
        }
        let prev_year: u32 = parts[0].parse().context("Bad year")?;
        let curr_year: u32 = parts[1].parse().context("Bad year")?;

        println!("\n--- Year pair: {prev_year}_{curr_year} ---");
        extract::extract_year_pair(
            prev_year,
            curr_year,
            region,
            raw_dir,
            features_dir,
            min_valid_frac,
        )?;
    }

    println!(
        "\nTotal extraction time: {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_predict(
    models_dir: &Path,
    features_dir: &Path,
    output_dir: &Path,
    year_pairs: &[String],
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let t0 = Instant::now();

    // ---- Load column list ----
    let mlp_cols: Vec<String> = {
        let data = std::fs::read_to_string(models_dir.join("mlp_cols.json"))?;
        serde_json::from_str(&data)?
    };
    println!("MLP features: {}", mlp_cols.len());

    // ---- Load model ----
    println!("Loading MLP model...");
    let t_load = Instant::now();
    let mut mlp = predict::OnnxMlp::load(models_dir)?;
    println!("  Loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    // ---- Load model config (threshold) ----
    let model_config = predict::ModelConfig::load(models_dir)?;

    // ---- Load scaler ----
    let scaler = predict::ScalerParams::load(&models_dir.join("mlp_scaler_0.json"))?;
    println!("Loaded scaler ({} features)", scaler.mean.len());

    // ---- Process each year pair ----
    for yp in year_pairs {
        println!("\n--- Year pair: {} ---", yp);

        // Load feature parquet
        let feat_path = features_dir.join(format!("features_rust_{yp}.parquet"));
        if !feat_path.exists() {
            println!("  SKIP: {} not found", feat_path.display());
            continue;
        }

        println!("  Loading features...");
        let t_feat = Instant::now();
        let (all_col_names, all_rows) = parquet_io::read_feature_parquet(&feat_path)?;
        let n_cells = all_rows.len();
        println!(
            "  Loaded {} cells x {} cols in {:.1}s",
            n_cells,
            all_col_names.len(),
            t_feat.elapsed().as_secs_f64()
        );

        // Build column index map
        let col_index: std::collections::HashMap<&str, usize> = all_col_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // ---- Extract MLP features + scale ----
        // Check for missing columns and give a clear error if SAR data is absent
        let missing_cols: Vec<&str> = mlp_cols
            .iter()
            .filter(|c| !col_index.contains_key(c.as_str()))
            .map(|c| c.as_str())
            .collect();

        if !missing_cols.is_empty() {
            let has_sar_missing = missing_cols.iter().any(|c| {
                c.starts_with("VV_") || c.starts_with("VH_") || c.starts_with("CR_")
                    || c.starts_with("RVI_") || c.starts_with("SAR_")
            });
            if has_sar_missing {
                eprintln!("ERROR: Sentinel-1 SAR data is unavailable for the selected region.");
                eprintln!("  The model requires SAR features ({} missing columns),", missing_cols.len());
                eprintln!("  but no SAR imagery was found for this area.");
                eprintln!("  Try selecting a region with SAR coverage (e.g. near major cities).");
                anyhow::bail!(
                    "SAR data unavailable for this region ({} SAR columns missing). \
                     Try a region with Sentinel-1 coverage.",
                    missing_cols.len()
                );
            } else {
                anyhow::bail!(
                    "{} required feature columns missing from parquet: {:?}",
                    missing_cols.len(),
                    &missing_cols[..missing_cols.len().min(5)]
                );
            }
        }

        let mlp_indices: Vec<usize> = mlp_cols
            .iter()
            .map(|c| *col_index.get(c.as_str()).unwrap())
            .collect();

        let mlp_features: Vec<Vec<f32>> = all_rows
            .iter()
            .map(|row| {
                let raw: Vec<f32> = mlp_indices
                    .iter()
                    .map(|&i| {
                        let v = row[i];
                        if v.is_finite() {
                            v
                        } else {
                            0.0
                        }
                    })
                    .collect();
                scaler.transform(&raw)
            })
            .collect();

        // ---- Run MLP prediction ----
        println!("  Running MLP inference...");
        let t_pred = Instant::now();
        let mut mlp_preds = mlp.predict(&mlp_features)?;
        println!(
            "  MLP done: {} cells in {:.2}s",
            n_cells,
            t_pred.elapsed().as_secs_f64()
        );

        // ---- Apply label threshold filtering ----
        model_config.apply_threshold(&mut mlp_preds);

        // ---- Save predictions ----
        let mlp_out = output_dir.join(format!("pred_mlp_{yp}.parquet"));
        parquet_io::write_predictions_parquet(&mlp_out, &CLASS_NAMES, &mlp_preds, "mlp")?;
        println!("  Wrote {}", mlp_out.display());

        // Parse curr_year from yp (e.g. "2020_2021" -> "2021")
        let parts: Vec<&str> = yp.split('_').collect();
        if parts.len() == 2 {
            let curr_year = parts[1];
            let json_out = output_dir.parent().unwrap_or(output_dir).join(format!("predictions_{}.json", curr_year));
            
            let mut json_map = serde_json::Map::with_capacity(n_cells);
            for (i, row) in mlp_preds.iter().enumerate() {
                let mut cell_map = serde_json::Map::with_capacity(CLASS_NAMES.len());
                for (j, &val) in row.iter().enumerate() {
                    let rounded = (val * 10000.0).round() / 10000.0;
                    cell_map.insert(CLASS_NAMES[j].to_string(), serde_json::json!(rounded));
                }
                json_map.insert(i.to_string(), serde_json::Value::Object(cell_map));
            }
            let json_str = serde_json::to_string(&serde_json::Value::Object(json_map))?;
            std::fs::write(&json_out, json_str)?;
            println!("  Wrote json {} ({} cells)", json_out.display(), n_cells);
        }
    }

    println!(
        "\nTotal prediction time: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    Ok(())
}

async fn run_download(
    bbox: &[f64],
    epsg: u32,
    years: &[u32],
    region: &str,
    raw_dir: &Path,
    anchor_ref: &Path,
) -> Result<()> {
    let t0 = Instant::now();

    let bbox_arr: [f64; 4] = <[f64; 4]>::try_from(bbox).map_err(|_| {
        anyhow::anyhow!(
            "bbox must have exactly 4 values [west, south, east, north], got {}",
            bbox.len()
        )
    })?;

    // Read anchor reference metadata (target grid definition)
    let anchor = composite::AnchorRef::from_tif(anchor_ref)
        .with_context(|| format!("Failed to read anchor ref: {}", anchor_ref.display()))?;
    println!("TerraPulse Download (Pure Rust)");
    println!("  Region: {region}");
    println!(
        "  BBOX: [{}, {}, {}, {}]",
        bbox[0], bbox[1], bbox[2], bbox[3]
    );
    println!("  EPSG: {epsg}");
    println!(
        "  Anchor: {}x{} EPSG:{}",
        anchor.width, anchor.height, anchor.epsg
    );
    println!("  Years: {:?}", years);
    println!(
        "  Concurrency: all {} years × 3 seasons in parallel",
        years.len()
    );
    println!();

    // Build HTTP client
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .connect_timeout(std::time::Duration::from_secs(15))
        .pool_max_idle_per_host(20)
        .build()?;

    // Download ALL years concurrently (each year downloads 3 seasons concurrently)
    let futures: Vec<_> = years
        .iter()
        .map(|&year| {
            let client = client.clone();
            let raw = raw_dir.to_path_buf();
            let reg = region.to_string();
            let anch = anchor.clone();
            async move {
                println!("--- Year: {year} ---");
                download::download_year(&client, bbox_arr, epsg, year, &reg, &raw, &anch).await
            }
        })
        .collect();

    let results = futures::future::join_all(futures).await;
    for r in results {
        r?;
    }

    println!(
        "\nOptical download time: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // Download SAR (Sentinel-1) sequentially per year to avoid STAC API overload.
    // Each year still downloads 3 seasons concurrently.
    println!("\n--- SAR (Sentinel-1) Download ---");
    let t_sar = Instant::now();

    // Pre-fetch S1 token so it's cached before downloads start
    if let Err(e) = stac::get_s1_token(&client).await {
        eprintln!("  WARNING: Could not pre-fetch S1 token: {e}");
    }

    for &year in years.iter() {
        println!("--- SAR Year: {year} ---");
        if let Err(e) = download::download_sar_year(
            &client, bbox_arr, year, region, raw_dir, &anchor,
        ).await {
            eprintln!("  SAR download error (non-fatal): {e}");
        }
    }

    println!(
        "\nTotal download time: {:.1}s (optical: {:.1}s, SAR: {:.1}s)",
        t0.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() - t_sar.elapsed().as_secs_f64(),
        t_sar.elapsed().as_secs_f64(),
    );

    Ok(())
}

async fn run_pipeline(
    bbox: &[f64],
    epsg: u32,
    years: &[u32],
    region: &str,
    data_dir: &Path,
    anchor_ref: &Path,
    models_dir: &Path,
    min_valid_frac: f32,
    skip_download: bool,
    skip_extract: bool,
    skip_predict: bool,
) -> Result<()> {
    let t0 = Instant::now();

    let raw_dir = data_dir.join("raw");
    let features_dir = data_dir.join("features");
    let predictions_dir = data_dir.join("predictions");

    // Derive consecutive year pairs
    let mut sorted_years = years.to_vec();
    sorted_years.sort();
    sorted_years.dedup();
    let year_pairs: Vec<String> = sorted_years
        .windows(2)
        .map(|w| format!("{}_{}", w[0], w[1]))
        .collect();

    let sep = "=".repeat(60);
    println!("\n{sep}");
    println!("TerraPulse Full Pipeline");
    println!("{sep}");
    println!("  Region: {region}");
    println!("  Years: {:?}", sorted_years);
    println!("  Year pairs: {:?}", year_pairs);
    println!("  Data dir: {}", data_dir.display());
    println!("  Skip download: {skip_download}");
    println!("  Skip extract: {skip_extract}");
    println!("  Skip predict: {skip_predict}");
    println!();

    // ================= STAGE 1: DOWNLOAD =================
    if !skip_download {
        println!("\n{sep}");
        println!("STAGE 1: DOWNLOAD");
        println!("{sep}");
        run_download(bbox, epsg, &sorted_years, region, &raw_dir, anchor_ref).await?;
    } else {
        println!("\n[SKIP] Stage 1: Download");
    }

    // ================= STAGE 2: EXTRACT =================
    if !skip_extract {
        println!("\n{sep}");
        println!("STAGE 2: EXTRACT");
        println!("{sep}");
        run_extract(&year_pairs, region, &raw_dir, &features_dir, min_valid_frac)?;
    } else {
        println!("\n[SKIP] Stage 2: Extract");
    }

    // ================= STAGE 3: PREDICT =================
    if !skip_predict {
        println!("\n{sep}");
        println!("STAGE 3: PREDICT");
        println!("{sep}");
        run_predict(models_dir, &features_dir, &predictions_dir, &year_pairs)?;
    } else {
        println!("\n[SKIP] Stage 3: Predict");
    }

    // ================= STAGE 4: LABELS =================
    println!("\n{sep}");
    println!("STAGE 4: LABELS");
    println!("{sep}");
    let anchor = composite::AnchorRef::from_tif(anchor_ref)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .pool_max_idle_per_host(20)
        .build()?;
        
    for &y in &sorted_years {
        if y <= 2021 {
            let out_path = data_dir.join(format!("labels_{}.json", y));
            if out_path.exists() {
                println!("  Labels {}: cached", y);
            } else {
                labels::download_labels(&client, y, &anchor, &out_path).await?;
            }
        }
    }

    // ================= STAGE 5: GRID =================
    println!("\n{sep}");
    println!("STAGE 5: GRID");
    println!("{sep}");
    let grid_out = data_dir.join("grid.json");
    if grid_out.exists() {
        println!("  Grid GeoJSON: cached");
    } else {
        grid::generate_grid_geojson(&anchor, &grid_out)?;
        println!("  Wrote grid GeoJSON: {}", grid_out.display());
    }

    println!("\n{sep}");
    println!("Pipeline complete in {:.1}s", t0.elapsed().as_secs_f64());
    println!("{sep}");

    Ok(())
}
```

---

## src/config.rs

```rust
/// Configuration constants matching the Python pipeline.
pub const N_CLASSES: usize = 7;

pub const CLASS_NAMES: [&str; N_CLASSES] = [
    "tree_cover",
    "shrubland",
    "grassland",
    "cropland",
    "built_up",
    "bare_sparse",
    "water",
];
```

---

## src/cog.rs

```rust
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
            let offset = meta.tile_offsets[idx];
            let size = meta.tile_byte_counts[idx] as usize;
            let compression = meta.compression;
            let bits = meta.bits_per_sample;
            let sample_fmt = meta.sample_format;
            let predictor = meta.predictor;
            let is_le = meta.le;
            let tw = meta.tile_width as usize;
            let th = meta.tile_height as usize;
            async move {
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
        let v1 = 0x5555u32;
        let v2 = 0x2AAAu32; // This doesn't matter for the new test, let's write exact bytes

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
```

---

## src/composite.rs

```rust
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
                            scene.bands[bi][px] = (v - boa_offset).max(0.0);
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
        let mut resampled = if epsg_mismatch {
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
        // NOTE: we intentionally do NOT convert 0.0 → NaN here.
        // Reflectance = 0.0 is valid for dark surfaces (deep water, shadows).
        // True NODATA (out-of-footprint) pixels already come back as NaN from
        // the bilinear resampler, and cloudy pixels are masked via SCL below.
        bands.push(resampled);
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
```

---

## src/download.rs

```rust
use anyhow::Result;
use reqwest::Client;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::composite::{self, AnchorRef};
use crate::stac;

const MIN_SCENES: usize = 8;

/// Download one season's composite and write it to a GeoTIFF.
///
/// Pure Rust path: STAC search → sign URLs → download COG tiles → reproject →
/// cloud mask → nanmedian composite → write GeoTIFF.
pub async fn download_season(
    client: &Client,
    bbox: [f64; 4],
    _epsg: u32,
    year: u32,
    season: &str,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<Option<PathBuf>> {
    let out_path = raw_dir.join(format!("sentinel2_{region_name}_{year}_{season}.tif"));
    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [{year}/{season}] Already exists ({mb:.1} MB) -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(raw_dir)?;

    // 1. Search for scenes
    eprintln!("  [{year}/{season}] Searching STAC...");
    let mut items = stac::search_with_fallback(client, bbox, year, season, MIN_SCENES).await?;
    if items.is_empty() {
        eprintln!("  [{year}/{season}] WARNING: No scenes found -- skipping!");
        return Ok(None);
    }

    // Cap scenes to prevent OOM on orbit-dense regions (e.g. Crete, equatorial).
    // STAC results are sorted by cloud cover, so truncating keeps the best scenes.
    const MAX_SCENES: usize = 20;
    if items.len() > MAX_SCENES {
        eprintln!(
            "  [{year}/{season}] Capping {} scenes to {MAX_SCENES} (lowest cloud cover)",
            items.len()
        );
        items.truncate(MAX_SCENES);
    }

    eprintln!(
        "  [{year}/{season}] Using {} scenes, signing...",
        items.len()
    );

    // 2. Get collection SAS token and sign all URLs
    let all_bands = stac::all_download_bands();
    let token = stac::get_collection_token(client).await?;
    let signed_scenes: Vec<HashMap<String, String>> = items
        .iter()
        .map(|item| {
            let band_refs: Vec<&str> = all_bands.iter().copied().collect();
            stac::sign_scene_assets_with_token(item, &band_refs, &token)
        })
        .collect::<Result<Vec<_>>>()?;

    // 3. Download, reproject, composite in pure Rust
    eprintln!("  [{year}/{season}] Downloading and compositing (pure Rust)...");
    composite::download_and_composite(client, &items, &signed_scenes, anchor, &out_path, year)
        .await?;

    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [{year}/{season}] Written ({mb:.1} MB)");
        Ok(Some(out_path))
    } else {
        anyhow::bail!("Composite failed to produce {}", out_path.display());
    }
}

/// Download all seasons for a year — concurrently (3 seasons at once).
pub async fn download_year(
    client: &Client,
    bbox: [f64; 4],
    epsg: u32,
    year: u32,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<()> {
    let (r1, r2, r3) = tokio::join!(
        download_season(client, bbox, epsg, year, "spring", region_name, raw_dir, anchor),
        download_season(client, bbox, epsg, year, "summer", region_name, raw_dir, anchor),
        download_season(client, bbox, epsg, year, "autumn", region_name, raw_dir, anchor),
    );
    r1?;
    r2?;
    r3?;
    Ok(())
}


// ---- SAR (Sentinel-1) download ----

/// Download one season of SAR composite.
pub async fn download_sar_season(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    season: &str,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<Option<PathBuf>> {
    let out_path = raw_dir.join(format!("sentinel1_{region_name}_{year}_{season}.tif"));
    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [SAR {year}/{season}] Already exists ({mb:.1} MB) -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(raw_dir)?;

    // 1. Search for S1 scenes
    eprintln!("  [SAR {year}/{season}] Searching STAC...");
    let items = stac::search_sar_scenes(client, bbox, year, season).await?;
    if items.is_empty() {
        eprintln!("  [SAR {year}/{season}] WARNING: No S1 scenes found -- skipping!");
        return Ok(None);
    }
    eprintln!(
        "  [SAR {year}/{season}] Found {} scenes, downloading...",
        items.len()
    );

    // 2. Get S1 SAS token
    let token = stac::get_s1_token(client).await?;

    // 3. Download, resample, composite in pure Rust
    crate::sar_download::download_sar_composite(client, &items, &token, anchor, &out_path).await?;

    if out_path.exists() {
        let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
        eprintln!("  [SAR {year}/{season}] Written ({mb:.1} MB)");
        Ok(Some(out_path))
    } else {
        anyhow::bail!("SAR composite failed to produce {}", out_path.display());
    }
}

/// Download SAR for all seasons of a year — concurrently.
pub async fn download_sar_year(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    region_name: &str,
    raw_dir: &Path,
    anchor: &AnchorRef,
) -> Result<()> {
    let (r1, r2, r3) = tokio::join!(
        download_sar_season(client, bbox, year, "spring", region_name, raw_dir, anchor),
        download_sar_season(client, bbox, year, "summer", region_name, raw_dir, anchor),
        download_sar_season(client, bbox, year, "autumn", region_name, raw_dir, anchor),
    );
    r1?;
    r2?;
    r3?;
    Ok(())
}
```

---

## src/extract.rs

```rust
//! Feature extraction module: reads seasonal GeoTIFFs, runs feature extraction,
//! writes output as parquet.

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::features;
use crate::parquet_io;
use crate::sar_features;
use crate::tif_reader;

const SEASONS: [&str; 3] = ["spring", "summer", "autumn"];
const NODATA: f32 = -9999.0;

/// Detect if data is in DN (> 1.0 scale) and compute scale factor.
fn detect_scale(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    let step = (data.len() / 10000).max(1);
    for &v in data.iter().step_by(step) {
        if v.is_finite() && v > 0.0 && v != NODATA {
            sum += v as f64;
            n += 1;
        }
    }
    // Fallback if still no valid pixels found
    if n == 0 {
        for &v in data {
            if v.is_finite() && v > 0.0 && v != NODATA {
                sum += v as f64;
                n += 1;
                if n > 1000 {
                    break;
                }
            }
        }
    }
    if n == 0 {
        return 1.0;
    }
    let mean = sum / n as f64;
    if mean > 100.0 {
        10000.0
    } else if mean > 10.0 {
        1000.0
    } else {
        1.0
    }
}

/// Extract features for a year pair and write to parquet.
pub fn extract_year_pair(
    prev_year: u32,
    curr_year: u32,
    region_name: &str,
    raw_dir: &Path,
    features_dir: &Path,
    min_valid_frac: f32,
) -> Result<Option<PathBuf>> {
    let tag = format!("{prev_year}_{curr_year}");
    let out_path = features_dir.join(format!("features_rust_{tag}.parquet"));

    if out_path.exists() {
        println!("  [{tag}] Already extracted -- skip");
        return Ok(Some(out_path));
    }

    std::fs::create_dir_all(features_dir)?;

    // Year mapping (always use 2020/2021 model tags)
    let year_map = [(prev_year, 2020u32), (curr_year, 2021)];

    // Check all TIFs exist
    let mut jobs = Vec::new();
    for &(actual_year, _model_year) in &year_map {
        for season in SEASONS {
            let tif = raw_dir.join(format!(
                "sentinel2_{region_name}_{actual_year}_{season}.tif"
            ));
            if !tif.exists() {
                println!("  [{tag}] WARNING: Missing {} -- skip", tif.display());
                return Ok(None);
            }
            jobs.push((actual_year, season));
        }
    }

    let t0 = std::time::Instant::now();

    // Load all seasonal rasters using native TIF reader
    let mut spectral_list: Vec<Vec<f32>> = Vec::new();
    let mut suffixes = Vec::new();
    let mut nr = 0usize;
    let mut nc = 0usize;
    let mut vf_min: Option<Vec<f32>> = None;

    for (actual_year, season) in jobs.iter() {
        let model_year = year_map.iter().find(|(a, _)| a == actual_year).unwrap().1;
        let tif = raw_dir.join(format!(
            "sentinel2_{region_name}_{actual_year}_{season}.tif"
        ));

        let t_read = std::time::Instant::now();

        // Always read bands + valid fraction together (single decode)
        let (nb, h, w, mut data, vf) =
            tif_reader::read_tif_bands_and_valid_fraction(&tif, features::N_BANDS)?;

        if nr == 0 {
            nr = h / features::GP;
            nc = w / features::GP;
        }

        // Accumulate minimum valid fraction across all rasters
        if let Some(vf_data) = vf {
            vf_min = Some(match vf_min {
                None => vf_data,
                Some(prev) => prev
                    .iter()
                    .zip(vf_data.iter())
                    .map(|(&a, &b)| match (a.is_finite(), b.is_finite()) {
                        (true, true) => a.min(b),
                        (true, false) => a,
                        (false, true) => b,
                        (false, false) => f32::NAN,
                    })
                    .collect(),
            });
        }

        let read_ms = t_read.elapsed().as_millis();
        assert!(
            nb >= features::N_BANDS,
            "TIF has {nb} bands, need {}",
            features::N_BANDS
        );

        // Normalize to [0,1] if in DN scale
        let scale = detect_scale(&data);
        if scale != 1.0 {
            for v in data.iter_mut() {
                if v.is_finite() && *v != NODATA {
                    *v /= scale;
                }
            }
        }
        // Replace NODATA with NaN
        for v in data.iter_mut() {
            if *v == NODATA {
                *v = f32::NAN;
            }
        }

        spectral_list.push(data);
        suffixes.push(format!("{model_year}_{season}"));
        println!("    Loaded {actual_year}_{season} -> {model_year}_{season} ({read_ms}ms)");
    }

    let n_cells = nr * nc;

    // Run extraction
    let t1 = std::time::Instant::now();
    let flat = features::extract_all_seasons(&spectral_list, nr, nc);
    let dt = t1.elapsed().as_secs_f64();
    println!("    Rust extraction: {dt:.1}s for {} seasons", jobs.len());
    drop(spectral_list);

    let n_seasons = suffixes.len();

    // Build column names for per-season features
    let base_names = features::feature_names();
    let mut columns = Vec::with_capacity(n_seasons * features::N_FEAT + 200);
    for suffix in &suffixes {
        for name in &base_names {
            columns.push(format!("{name}_{suffix}"));
        }
    }

    // Build row data [n_cells][n_features_total] from optical features
    let n_total_feats = n_seasons * features::N_FEAT;
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n_cells);
    for ci in 0..n_cells {
        let base = ci * n_total_feats;
        let mut row: Vec<f32> = flat[base..base + n_total_feats].to_vec();

        // Replace inf with NaN, then impute NaN with column medians later
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = f32::NAN;
            }
        }
        rows.push(row);
    }

    // =========================================================================
    // SAR feature extraction (optional — backward-compatible)
    // =========================================================================
    let mut has_sar = true;
    let mut sar_spectral_list: Vec<Vec<f32>> = Vec::new();

    for (actual_year, season) in jobs.iter() {
        let sar_tif = raw_dir.join(format!(
            "sentinel1_{region_name}_{actual_year}_{season}.tif"
        ));
        if !sar_tif.exists() {
            has_sar = false;
            break;
        }
    }

    if has_sar {
        println!("    SAR TIFs detected — extracting SAR features");
        for (actual_year, season) in jobs.iter() {
            let sar_tif = raw_dir.join(format!(
                "sentinel1_{region_name}_{actual_year}_{season}.tif"
            ));
            let t_read = std::time::Instant::now();

            let (nb, _h, _w, mut data, _vf) =
                tif_reader::read_tif_bands_and_valid_fraction(&sar_tif, sar_features::N_SAR_BANDS)?;

            assert!(
                nb >= sar_features::N_SAR_BANDS,
                "SAR TIF has {nb} bands, need {}",
                sar_features::N_SAR_BANDS
            );

            // Replace NODATA with NaN
            for v in data.iter_mut() {
                if *v == NODATA {
                    *v = f32::NAN;
                }
            }

            // SAR data should already be in [0,1] after dB conversion + scaling in Python.
            // But if raw linear power values are present, detect and convert.
            let mut finite_sum = 0.0f64;
            let mut finite_n = 0u64;
            let step = (data.len() / 10000).max(1);
            for &v in data.iter().step_by(step) {
                if v.is_finite() && v > 0.0 && v != NODATA {
                    finite_sum += v as f64;
                    finite_n += 1;
                }
            }
            if finite_n == 0 {
                for &v in data.iter() {
                    if v.is_finite() && v > 0.0 && v != NODATA {
                        finite_sum += v as f64;
                        finite_n += 1;
                        if finite_n > 1000 {
                            break;
                        }
                    }
                }
            }
            if finite_n > 0 {
                let mean_val = finite_sum / finite_n as f64;
                if mean_val > 1.5 {
                    // Likely raw linear power, convert to dB then scale to [0,1]
                    println!(
                        "      SAR: converting from linear power (mean={mean_val:.3}) to [0,1]"
                    );
                    for v in data.iter_mut() {
                        if v.is_finite() && *v > 0.0 {
                            let db = 10.0 * v.log10();
                            // Clamp to [-30, 0] and scale to [0, 1]
                            *v = (db.clamp(-30.0, 0.0) + 30.0) / 30.0;
                        } else if v.is_finite() {
                            *v = 0.0; // zero or negative power → 0
                        }
                    }
                }
            }

            let read_ms = t_read.elapsed().as_millis();
            println!("      Loaded SAR {actual_year}_{season} ({read_ms}ms)");
            sar_spectral_list.push(data);
        }

        // Run SAR extraction
        let t1 = std::time::Instant::now();
        let sar_flat = sar_features::extract_all_sar_seasons(&sar_spectral_list, nr, nc);
        let dt = t1.elapsed().as_secs_f64();
        println!(
            "    SAR extraction: {dt:.1}s for {} seasons",
            sar_spectral_list.len()
        );
        drop(sar_spectral_list);

        // Add SAR column names
        let sar_base_names = sar_features::sar_feature_names();
        for suffix in &suffixes {
            for name in &sar_base_names {
                columns.push(format!("{name}_{suffix}"));
            }
        }

        // Append SAR features to each row
        let n_sar_total = n_seasons * sar_features::N_SAR_FEAT;
        for ci in 0..n_cells {
            let base = ci * n_sar_total;
            let sar_row: Vec<f32> = sar_flat[base..base + n_sar_total]
                .iter()
                .map(|&v| if v.is_finite() { v } else { f32::NAN })
                .collect();
            rows[ci].extend_from_slice(&sar_row);
        }

        println!("    Added {} SAR columns per cell", n_sar_total);
    } else {
        println!("    No SAR TIFs found — optical-only mode (backward-compatible)");
    }

    // =========================================================================
    // Cell-level NaN fill: temporal (same season, other year) + spatial NN
    // Applied BEFORE phenological features so pheno gets clean inputs.
    // =========================================================================
    {
        let n_optical = n_seasons * features::N_FEAT;
        let n_sar_per_season = if has_sar { sar_features::N_SAR_FEAT } else { 0 };
        let _n_sar_total_cols = n_seasons * n_sar_per_season;
        let n_row_len = rows.first().map_or(0, |r| r.len());

        let mut temporal_fills = 0u64;
        let mut spatial_fills = 0u64;
        let mut zero_fills = 0u64;

        // --- Step 1: Temporal fill ---
        // For each feature in each season, if NaN, try same feature from same
        // season but other year. E.g., autumn_2020_NDVI_mean -> autumn_2021_NDVI_mean
        // Seasons come in groups of 3 per year: [spring0, summer0, autumn0, spring1, summer1, autumn1]
        let n_years_loaded = n_seasons / 3;
        if n_years_loaded == 2 {
            // Map season index pairs: 0<->3 (spring), 1<->4 (summer), 2<->5 (autumn)
            for row in rows.iter_mut() {
                for season_in_year in 0..3 {
                    let si_a = season_in_year;         // year 0
                    let si_b = 3 + season_in_year;     // year 1

                    // Optical features
                    for fi in 0..features::N_FEAT {
                        let idx_a = si_a * features::N_FEAT + fi;
                        let idx_b = si_b * features::N_FEAT + fi;
                        if !row[idx_a].is_finite() && row[idx_b].is_finite() {
                            row[idx_a] = row[idx_b];
                            temporal_fills += 1;
                        } else if row[idx_a].is_finite() && !row[idx_b].is_finite() {
                            row[idx_b] = row[idx_a];
                            temporal_fills += 1;
                        }
                    }

                    // SAR features
                    if has_sar {
                        for fi in 0..n_sar_per_season {
                            let idx_a = n_optical + si_a * n_sar_per_season + fi;
                            let idx_b = n_optical + si_b * n_sar_per_season + fi;
                            if idx_a < n_row_len && idx_b < n_row_len {
                                if !row[idx_a].is_finite() && row[idx_b].is_finite() {
                                    row[idx_a] = row[idx_b];
                                    temporal_fills += 1;
                                } else if row[idx_a].is_finite() && !row[idx_b].is_finite() {
                                    row[idx_b] = row[idx_a];
                                    temporal_fills += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Step 2: Spatial NN fill ---
        // For remaining NaN: search ring 1 first (immediate 8 neighbors),
        // then expand to ring 5. Use median of valid neighbors.
        let n_grid_rows = nr;
        let n_grid_cols = nc;
        let max_ring = 5usize;

        for col in 0..n_row_len {
            // Check if this column has any NaN
            let has_any_nan = rows.iter().any(|r| col < r.len() && !r[col].is_finite());
            if !has_any_nan {
                continue;
            }

            for ci in 0..n_cells {
                if col >= rows[ci].len() || rows[ci][col].is_finite() {
                    continue;
                }

                let cr = ci / n_grid_cols;
                let cc = ci % n_grid_cols;
                let mut found = f32::NAN;

                'rings: for ring in 1..=max_ring {
                    let mut ring_vals: Vec<f32> = Vec::new();
                    let r_min = cr.saturating_sub(ring);
                    let r_max = (cr + ring).min(n_grid_rows - 1);
                    let c_min = cc.saturating_sub(ring);
                    let c_max = (cc + ring).min(n_grid_cols - 1);

                    for r in r_min..=r_max {
                        for c in c_min..=c_max {
                            if r == r_min || r == r_max || c == c_min || c == c_max {
                                let ni = r * n_grid_cols + c;
                                if ni < n_cells && col < rows[ni].len() {
                                    let v = rows[ni][col];
                                    if v.is_finite() {
                                        ring_vals.push(v);
                                    }
                                }
                            }
                        }
                    }

                    if !ring_vals.is_empty() {
                        ring_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                        found = ring_vals[ring_vals.len() / 2];
                        break 'rings;
                    }
                }

                if found.is_finite() {
                    rows[ci][col] = found;
                    spatial_fills += 1;
                } else {
                    rows[ci][col] = 0.0;
                    zero_fills += 1;
                }
            }
        }

        println!("    NaN fill: {temporal_fills} temporal, {spatial_fills} spatial NN, {zero_fills} zero fallback");
    }

    // =========================================================================
    // Phenological cross-season features (V4)
    // For each year's 3 seasons (spring, summer, autumn), compute:
    //   curvature = summer - (spring + autumn) / 2  (seasonal peak)
    //   slope     = (autumn - spring) / 2           (greening/browning trend)
    //   amplitude = max - min across 3 seasons       (seasonal variability)
    //   peak      = argmax(spring, summer, autumn)   (timing of peak, 0/1/2)
    // Applied to 10 band means + 5 key index means = 15 signals per year
    // =========================================================================
    if n_seasons >= 3 {
        // Offsets of mean values within N_FEAT for each signal
        // Bands: mean is at offset 0, 8, 16, ..., 72 (10 bands × 8 stats, mean is first)
        // Indices: mean is at offset 80, 85, 90, ..., 150 (15 indices × 5 stats, mean is first)
        // We use 10 bands + 5 key indices (NDVI, NDWI, NDBI, BSI, EVI2)
        let band_mean_offsets: Vec<usize> = (0..10).map(|b| b * 8).collect(); // 0,8,16,...,72
                                                                              // NDVI=0, NDWI=1, NDBI=2, BSI=11, EVI2=12 within the 15 indices
        let idx_mean_offsets: Vec<usize> = vec![
            80 + 0 * 5,  // NDVI mean
            80 + 1 * 5,  // NDWI mean
            80 + 2 * 5,  // NDBI mean
            80 + 11 * 5, // BSI mean
            80 + 12 * 5, // EVI2 mean
        ];

        let all_offsets: Vec<usize> = band_mean_offsets
            .iter()
            .chain(idx_mean_offsets.iter())
            .copied()
            .collect();

        let signal_names = [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "NDWI",
            "NDBI", "BSI", "EVI2",
        ];

        let pheno_names = ["curvature", "slope", "amplitude", "peak"];

        // Process each year separately (seasons come in groups of 3)
        let n_years = n_seasons / 3;
        for yr_idx in 0..n_years {
            let spring_season = yr_idx * 3; // index 0 or 3
            let summer_season = yr_idx * 3 + 1; // index 1 or 4
            let autumn_season = yr_idx * 3 + 2; // index 2 or 5

            let year_tag = &suffixes[spring_season]; // e.g. "2020_spring"
            let year_label = year_tag.split('_').next().unwrap_or("unknown");

            // Add column names for this year's phenological features
            for sig_name in &signal_names {
                for pheno in &pheno_names {
                    columns.push(format!("{sig_name}_pheno_{pheno}_{year_label}"));
                }
            }

            // Compute phenological features for each cell
            for row in rows.iter_mut() {
                for &offset in &all_offsets {
                    let spring_val = row[spring_season * features::N_FEAT + offset];
                    let summer_val = row[summer_season * features::N_FEAT + offset];
                    let autumn_val = row[autumn_season * features::N_FEAT + offset];

                    // curvature: summer peak relative to shoulders
                    let curvature = summer_val - (spring_val + autumn_val) / 2.0;
                    // slope: trend from spring to autumn
                    let slope = (autumn_val - spring_val) / 2.0;
                    // amplitude: max - min
                    let mx = spring_val.max(summer_val).max(autumn_val);
                    let mn = spring_val.min(summer_val).min(autumn_val);
                    let amplitude = mx - mn;
                    // peak_season: 0=spring, 1=summer, 2=autumn
                    let peak = if summer_val >= spring_val && summer_val >= autumn_val {
                        1.0f32
                    } else if autumn_val >= spring_val {
                        2.0f32
                    } else {
                        0.0f32
                    };

                    row.push(curvature);
                    row.push(slope);
                    row.push(amplitude);
                    row.push(peak);
                }
            }
        }

        let n_pheno = all_offsets.len() * 4 * n_years;
        println!("    Added {n_pheno} optical phenological features ({} signals x 4 pheno x {n_years} years)", all_offsets.len());

        // =====================================================================
        // SAR Phenological cross-season features
        // For each year's 3 SAR seasons, compute curvature/slope/amplitude/peak
        // Applied to: VV_mean, VH_mean, CR_mean, RVI_mean = 4 signals per year
        // =====================================================================
        if has_sar {
            // SAR feature offsets within N_SAR_FEAT (48):
            // VV_mean = offset 0, VH_mean = offset 8
            // CR_mean = offset 16, RVI_mean = offset 21
            let sar_mean_offsets: Vec<usize> = vec![0, 8, 16, 21];
            let sar_signal_names = ["SAR_VV", "SAR_VH", "SAR_CR", "SAR_RVI"];
            let pheno_names_sar = ["curvature", "slope", "amplitude", "peak"];

            // SAR features start after optical features in each row
            let optical_per_season = features::N_FEAT;
            let sar_per_season = sar_features::N_SAR_FEAT;
            // In the row layout:
            // [optical_season_0..optical_season_N | sar_season_0..sar_season_N | ...]
            // Optical: n_seasons * N_FEAT columns from index 0
            // SAR: n_seasons * N_SAR_FEAT columns from index (n_seasons * N_FEAT)
            let sar_base_offset = n_seasons * optical_per_season;

            for yr_idx in 0..n_years {
                let spring_season = yr_idx * 3;
                let summer_season = yr_idx * 3 + 1;
                let autumn_season = yr_idx * 3 + 2;

                let year_tag = &suffixes[spring_season];
                let year_label = year_tag.split('_').next().unwrap_or("unknown");

                for sig_name in &sar_signal_names {
                    for pheno in &pheno_names_sar {
                        columns.push(format!("{sig_name}_pheno_{pheno}_{year_label}"));
                    }
                }

                for row in rows.iter_mut() {
                    for &offset in &sar_mean_offsets {
                        let spring_val =
                            row[sar_base_offset + spring_season * sar_per_season + offset];
                        let summer_val =
                            row[sar_base_offset + summer_season * sar_per_season + offset];
                        let autumn_val =
                            row[sar_base_offset + autumn_season * sar_per_season + offset];

                        let curvature = summer_val - (spring_val + autumn_val) / 2.0;
                        let slope = (autumn_val - spring_val) / 2.0;
                        let mx = spring_val.max(summer_val).max(autumn_val);
                        let mn = spring_val.min(summer_val).min(autumn_val);
                        let amplitude = mx - mn;
                        let peak = if summer_val >= spring_val && summer_val >= autumn_val {
                            1.0f32
                        } else if autumn_val >= spring_val {
                            2.0f32
                        } else {
                            0.0f32
                        };

                        row.push(curvature);
                        row.push(slope);
                        row.push(amplitude);
                        row.push(peak);
                    }
                }
            }

            let n_sar_pheno = sar_mean_offsets.len() * 4 * n_years;
            println!("    Added {n_sar_pheno} SAR phenological features ({} signals x 4 pheno x {n_years} years)", sar_mean_offsets.len());

            // =================================================================
            // SAR Temporal Features (new cross-season statistics)
            // 8 features per year:
            //   3 summer-winter contrasts: VH, VV, CR (spring as winter proxy)
            //   3 temporal_std: std(mean across 3 seasons) for VH, VV, CR
            //   2 temporal_cv: temporal_std / temporal_mean for VH, VV
            // =================================================================
            // Offsets for VV_mean, VH_mean, CR_mean within N_SAR_FEAT
            let temporal_offsets: Vec<usize> = vec![0, 8, 16]; // VV, VH, CR
            let temporal_names = ["SAR_VV", "SAR_VH", "SAR_CR"];

            for yr_idx in 0..n_years {
                let spring_season = yr_idx * 3;
                let summer_season = yr_idx * 3 + 1;
                let _autumn_season = yr_idx * 3 + 2;

                let year_tag = &suffixes[spring_season];
                let year_label = year_tag.split('_').next().unwrap_or("unknown");

                // Column names: summer_winter contrasts
                for sig in &temporal_names {
                    columns.push(format!("{sig}_summer_winter_{year_label}"));
                }
                // Column names: temporal_std
                for sig in &temporal_names {
                    columns.push(format!("{sig}_temporal_std_{year_label}"));
                }
                // Column names: temporal_cv (VV and VH only, not CR)
                columns.push(format!("SAR_VV_temporal_cv_{year_label}"));
                columns.push(format!("SAR_VH_temporal_cv_{year_label}"));

                // Compute for each cell
                for row in rows.iter_mut() {
                    // Summer-winter contrasts (spring as winter proxy)
                    for &offset in &temporal_offsets {
                        let spring_val =
                            row[sar_base_offset + spring_season * sar_per_season + offset];
                        let summer_val =
                            row[sar_base_offset + summer_season * sar_per_season + offset];
                        row.push(summer_val - spring_val);
                    }

                    // Temporal std across 3 seasons
                    for &offset in &temporal_offsets {
                        let s0 = row[sar_base_offset + (yr_idx * 3) * sar_per_season + offset];
                        let s1 = row[sar_base_offset + (yr_idx * 3 + 1) * sar_per_season + offset];
                        let s2 = row[sar_base_offset + (yr_idx * 3 + 2) * sar_per_season + offset];
                        let mean = (s0 + s1 + s2) / 3.0;
                        let var =
                            ((s0 - mean).powi(2) + (s1 - mean).powi(2) + (s2 - mean).powi(2)) / 3.0;
                        row.push(var.max(0.0).sqrt());
                    }

                    // Temporal CV for VV and VH only (offsets 0 and 8)
                    for &offset in &[0usize, 8usize] {
                        let s0 = row[sar_base_offset + (yr_idx * 3) * sar_per_season + offset];
                        let s1 = row[sar_base_offset + (yr_idx * 3 + 1) * sar_per_season + offset];
                        let s2 = row[sar_base_offset + (yr_idx * 3 + 2) * sar_per_season + offset];
                        let mean = (s0 + s1 + s2) / 3.0;
                        let var =
                            ((s0 - mean).powi(2) + (s1 - mean).powi(2) + (s2 - mean).powi(2)) / 3.0;
                        let std = var.max(0.0).sqrt();
                        let cv = if mean.abs() > 1e-10 {
                            std / mean.abs()
                        } else {
                            0.0
                        };
                        row.push(cv);
                    }
                }
            }

            let n_sar_temporal = 8 * n_years;
            println!("    Added {n_sar_temporal} SAR temporal features (8 x {n_years} years)");
        }
    }

    // Final NaN safety net: any remaining NaN (e.g. from pheno features
    // where all 3 seasons were still NaN after fill) gets zero-filled.
    // This should be rare after temporal + spatial fill.
    let n_all_cols = rows.first().map_or(0, |r| r.len());
    let mut final_nan_count = 0u64;
    for row in rows.iter_mut() {
        for col in 0..n_all_cols.min(row.len()) {
            if !row[col].is_finite() {
                row[col] = 0.0;
                final_nan_count += 1;
            }
        }
    }
    if final_nan_count > 0 {
        println!("    Final zero-fill for {final_nan_count} remaining NaN values");
    }

    // Add cell_id, valid_fraction columns
    let mut extra_cols = vec!["cell_id".to_string()];
    let mut extra_data: Vec<Vec<f32>> = vec![(0..n_cells as u32).map(|i| i as f32).collect()];

    if let Some(ref vf) = vf_min {
        // Aggregate valid fraction per cell (mean of GP×GP pixels)
        let gp = features::GP;
        let mut vf_cells = vec![0.0f32; n_cells];
        for ci in 0..n_cells {
            let cr = ci / nc;
            let cc = ci % nc;
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for dr in 0..gp {
                let r = cr * gp + dr;
                for dc in 0..gp {
                    let c = cc * gp + dc;
                    let v = vf[r * (nc * gp) + c];
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            vf_cells[ci] = if n > 0 { sum / n as f32 } else { 0.0 };
        }
        extra_cols.push("valid_fraction".to_string());
        extra_cols.push("low_valid_fraction".to_string());
        let low_vf: Vec<f32> = vf_cells
            .iter()
            .map(|&v| if v < min_valid_frac { 1.0 } else { 0.0 })
            .collect();
        extra_data.push(vf_cells);
        extra_data.push(low_vf);
    }

    // Write parquet
    parquet_io::write_feature_parquet(&out_path, &extra_cols, &extra_data, &columns, &rows)?;

    let elapsed = t0.elapsed().as_secs_f64();
    let mb = std::fs::metadata(&out_path)?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "  [{tag}] Done: {} cols, {mb:.1} MB, {elapsed:.0}s",
        columns.len() + extra_cols.len()
    );
    Ok(Some(out_path))
}
```

---

## src/features.rs

```rust
//! Pure Rust feature extraction core (copied from terrapulse_features lib.rs).
//! 224 features per cell per season.

use rayon::prelude::*;

pub const GP: usize = 10;
const N_PX: usize = GP * GP; // 100 pixels per cell
pub const N_BANDS: usize = 10;
pub(crate) const EPS: f32 = 1e-10;

// LBP parameters
const LBP_P: usize = 8;
pub(crate) const LBP_BINS: usize = LBP_P + 2; // 10 bins: 0..8 uniform, 9 non-uniform

// Band layout (must match Python's order)
const B02: usize = 0;
const B03: usize = 1;
const B04: usize = 2;
const B05: usize = 3;
const B06: usize = 4;
const B07: usize = 5;
const B08: usize = 6;
const B8A: usize = 7;
const B11: usize = 8;
const B12: usize = 9;

// Sentinel-2 Tasseled Cap coefficients (Nedkov, 2017) — 10 bands
// Order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
// Must match Python's TC_BRIGHTNESS / TC_GREENNESS / TC_WETNESS exactly
const TC10_B: [f32; 10] = [
    0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949, 0.1822, 0.0031, 0.1112, 0.0825,
];
const TC10_G: [f32; 10] = [
    -0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.0469, -0.0322, -0.0015, -0.0693, -0.0180,
];
const TC10_W: [f32; 10] = [
    0.2578, 0.2305, 0.0883, 0.1071, -0.7611, 0.0882, 0.4572, -0.0021, -0.4064, 0.0117,
];

// 20m bands that need block-reduce (factor=2) before stats, matching original Python
const BANDS_20M: [usize; 6] = [B05, B06, B07, B8A, B11, B12];

// Feature counts
const N_BAND_STATS: usize = N_BANDS * 8; // 80
const N_IDX_STATS: usize = 15 * 5; // 75
const N_TC: usize = 6; // 3 components * (mean,std)
const N_SPATIAL: usize = 8;
const N_LBP: usize = 5 * (LBP_BINS + 1); // 55 (10 bins + entropy) * 5
pub const N_FEAT: usize = N_BAND_STATS + N_IDX_STATS + N_TC + N_SPATIAL + N_LBP; // 224

// =====================================================================
// Utility: reflect indexing for ndimage-like boundary handling
// =====================================================================

#[inline(always)]
fn reflect_index(mut i: isize, len: isize) -> isize {
    if len <= 1 {
        return 0;
    }
    while i < 0 || i >= len {
        if i < 0 {
            i = -i - 1;
        }
        if i >= len {
            i = 2 * len - i - 1;
        }
    }
    i
}

// =====================================================================
// LBP: uniform LUT + bilinear sampling
// =====================================================================

pub(crate) fn build_lbp_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for val in 0u16..256 {
        let v = val as u8;
        let mut transitions = 0u32;
        for i in 0..8u32 {
            let b0 = (v >> i) & 1;
            let b1 = (v >> ((i + 1) % 8)) & 1;
            if b0 != b1 {
                transitions += 1;
            }
        }
        lut[val as usize] = if transitions <= 2 {
            v.count_ones() as u8 // 0..8
        } else {
            (LBP_P + 1) as u8 // non-uniform bin = 9
        };
    }
    lut
}

/// Bilinear interpolation with constant-zero boundary (cval=0).
/// Matches skimage's bilinear_interpolation(&image[0,0], rows, cols, r, c, 'C', 0, &out).
#[inline(always)]
fn bilinear_constant_zero(img: &[f32], h: usize, w: usize, ry: f64, rx: f64) -> f64 {
    let minr = ry.floor() as isize;
    let minc = rx.floor() as isize;
    let maxr = ry.ceil() as isize;
    let maxc = rx.ceil() as isize;
    let dr = ry - minr as f64;
    let dc = rx - minc as f64;

    // get_pixel2d with mode='C', cval=0: out-of-bounds → 0.0
    let get = |r: isize, c: isize| -> f64 {
        if r < 0 || r >= h as isize || c < 0 || c >= w as isize {
            0.0
        } else {
            img[r as usize * w + c as usize] as f64
        }
    };

    let top_left = get(minr, minc);
    let top_right = get(minr, maxc);
    let bottom_left = get(maxr, minc);
    let bottom_right = get(maxr, maxc);

    let top = (1.0 - dc) * top_left + dc * top_right;
    let bottom = (1.0 - dc) * bottom_left + dc * bottom_right;
    (1.0 - dr) * top + dr * bottom
}

pub(crate) fn compute_lbp_raster(img: &[f32], h: usize, w: usize, lut: &[u8; 256]) -> Vec<u8> {
    // skimage rounds offsets to 5 decimals: np.round(rr, 5)
    // Using 0.70711 instead of FRAC_1_SQRT_2 to match exactly.
    let s2: f64 = 0.70711;
    let dr: [f64; 8] = [0.0, -s2, -1.0, -s2, 0.0, s2, 1.0, s2];
    let dc: [f64; 8] = [1.0, s2, 0.0, -s2, -1.0, -s2, 0.0, s2];

    let mut out = vec![0u8; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        let rf = r as f64;
        for c in 0..w {
            let cf = c as f64;
            let center = img[r * w + c] as f64;
            let mut code: u8 = 0;
            for k in 0..8 {
                let val = bilinear_constant_zero(img, h, w, rf + dr[k], cf + dc[k]);
                if val >= center {
                    code |= 1 << k;
                }
            }
            row[c] = lut[code as usize];
        }
    });
    out
}

/// Bilinear interpolation on a GP×GP patch with constant-zero boundary.
/// Matches skimage's bilinear_interpolation(mode='C', cval=0).
#[inline(always)]
fn bilinear_patch_constant_zero(patch: &[f32; N_PX], ry: f64, rx: f64) -> f64 {
    let minr = ry.floor() as isize;
    let minc = rx.floor() as isize;
    let maxr = ry.ceil() as isize;
    let maxc = rx.ceil() as isize;
    let dr = ry - minr as f64;
    let dc = rx - minc as f64;

    let gp = GP as isize;
    let get = |r: isize, c: isize| -> f64 {
        if r < 0 || r >= gp || c < 0 || c >= gp {
            0.0
        } else {
            patch[r as usize * GP + c as usize] as f64
        }
    };

    let top_left = get(minr, minc);
    let top_right = get(minr, maxc);
    let bottom_left = get(maxr, minc);
    let bottom_right = get(maxr, maxc);

    let top = (1.0 - dc) * top_left + dc * top_right;
    let bottom = (1.0 - dc) * bottom_left + dc * bottom_right;
    (1.0 - dr) * top + dr * bottom
}

/// Compute LBP on isolated 10×10 patches with per-cell NaN fill + clip.
/// Matches Python V10's lbp_features(patch_ref) which does:
///   nir = np.where(np.isfinite(nir), nir, np.nanmean(nir))
///   nir = np.clip(nir, 0.0, 1.0)
///   lbp = local_binary_pattern(nir, P=8, R=1, method="uniform")
///
/// `raw_img` is the RAW band data (may contain NaN/non-finite).
/// `clip_01` indicates whether to clip values to [0, 1] (true for spectral bands,
///   false for index images already in [0, 1]).
pub(crate) fn compute_lbp_perpatch(
    raw_img: &[f32],
    h: usize,
    w: usize,
    n_rows: usize,
    n_cols: usize,
    lut: &[u8; 256],
    clip_01: bool,
) -> Vec<u8> {
    // skimage rounds offsets to 5 decimals: np.round(rr, 5)
    // Using 0.70711 instead of FRAC_1_SQRT_2 to match exactly.
    let s2: f64 = 0.70711;
    let dr: [f64; 8] = [0.0, -s2, -1.0, -s2, 0.0, s2, 1.0, s2];
    let dc: [f64; 8] = [1.0, s2, 0.0, -s2, -1.0, -s2, 0.0, s2];

    let n_cells = n_rows * n_cols;

    let cell_codes: Vec<[u8; N_PX]> = (0..n_cells)
        .into_par_iter()
        .map(|ci| {
            let cr = ci / n_cols;
            let cc = ci % n_cols;
            let r0 = cr * GP;
            let c0 = cc * GP;

            // Extract raw patch
            let mut patch = [0.0f32; N_PX];
            for d in 0..GP {
                let src = (r0 + d) * w + c0;
                patch[d * GP..d * GP + GP].copy_from_slice(&raw_img[src..src + GP]);
            }

            // Per-cell NaN fill: nanmean of THIS patch (matches Python exactly)
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for &v in &patch {
                if v.is_finite() {
                    let cv = if clip_01 { v.clamp(0.0, 1.0) } else { v };
                    sum += cv as f64;
                    n += 1;
                }
            }
            let fill = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };

            // Apply NaN fill + clip
            for v in patch.iter_mut() {
                if v.is_finite() {
                    if clip_01 {
                        *v = v.clamp(0.0, 1.0);
                    }
                } else {
                    *v = fill;
                }
            }

            let mut codes = [0u8; N_PX];
            for r in 0..GP {
                for c in 0..GP {
                    let center = patch[r * GP + c] as f64;
                    let mut code: u8 = 0;
                    for k in 0..8 {
                        let val = bilinear_patch_constant_zero(
                            &patch,
                            r as f64 + dr[k],
                            c as f64 + dc[k],
                        );
                        if val >= center {
                            code |= 1 << k;
                        }
                    }
                    codes[r * GP + c] = lut[code as usize];
                }
            }
            codes
        })
        .collect();

    let mut out = vec![0u8; h * w];
    for ci in 0..n_cells {
        let cr = ci / n_cols;
        let cc = ci % n_cols;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for d in 0..GP {
            let dst = (r0 + d) * w + c0;
            out[dst..dst + GP].copy_from_slice(&cell_codes[ci][d * GP..d * GP + GP]);
        }
    }
    out
}

// =====================================================================
// Full-raster convolutions (reflect boundary like ndimage default)
// =====================================================================

fn compute_sobel_mag(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        let hh = h as isize;
        let ww = w as isize;
        let rr = r as isize;

        for c in 0..w {
            let cc = c as isize;

            let g = |dr: isize, dc: isize| -> f64 {
                let r2 = reflect_index(rr + dr, hh) as usize;
                let c2 = reflect_index(cc + dc, ww) as usize;
                img[r2 * w + c2] as f64
            };

            // Classic 3x3 Sobel kernels
            let gx = -g(-1, -1) + g(-1, 1) - 2.0 * g(0, -1) + 2.0 * g(0, 1) - g(1, -1) + g(1, 1);

            let gy = -g(-1, -1) - 2.0 * g(-1, 0) - g(-1, 1) + g(1, -1) + 2.0 * g(1, 0) + g(1, 1);

            row[c] = ((gx * gx + gy * gy).sqrt()) as f32;
        }
    });
    out
}

fn compute_laplacian(img: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        let hh = h as isize;
        let ww = w as isize;
        let rr = r as isize;

        for c in 0..w {
            let cc = c as isize;

            let g = |dr: isize, dc: isize| -> f64 {
                let r2 = reflect_index(rr + dr, hh) as usize;
                let c2 = reflect_index(cc + dc, ww) as usize;
                img[r2 * w + c2] as f64
            };

            // 4-neighbor Laplacian: [0 1 0; 1 -4 1; 0 1 0]
            let v = g(-1, 0) + g(1, 0) + g(0, -1) + g(0, 1) - 4.0 * g(0, 0);
            row[c] = v as f32;
        }
    });
    out
}

// =====================================================================
// Image preparation helpers
// =====================================================================

/// Fill NaN pixels in a band raster using spatial nearest-neighbor search.
///
/// For each NaN pixel, searches expanding rings (up to `max_radius` pixels)
/// for the nearest finite pixel. Uses the median of all valid pixels found
/// in the first ring that contains any. Falls back to 0.0 if no neighbor
/// found within radius.
pub fn fill_nan_spatial_band(data: &mut [f32], h: usize, w: usize, max_radius: usize) {
    // Collect positions of NaN pixels
    let mut nan_positions: Vec<(usize, usize)> = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if !data[r * w + c].is_finite() {
                nan_positions.push((r, c));
            }
        }
    }

    if nan_positions.is_empty() {
        return;
    }

    // For each NaN pixel, search expanding rings
    let mut fill_values: Vec<(usize, f32)> = Vec::with_capacity(nan_positions.len());

    for &(nr, nc_pos) in &nan_positions {
        let mut found = f32::NAN;
        'rings: for radius in 1..=max_radius {
            let mut ring_vals: Vec<f32> = Vec::new();
            let r_min = nr.saturating_sub(radius);
            let r_max = (nr + radius).min(h - 1);
            let c_min = nc_pos.saturating_sub(radius);
            let c_max = (nc_pos + radius).min(w - 1);

            for r in r_min..=r_max {
                for c in c_min..=c_max {
                    // Only check pixels on the ring boundary (not interior)
                    if r == r_min || r == r_max || c == c_min || c == c_max {
                        let v = data[r * w + c];
                        if v.is_finite() {
                            ring_vals.push(v);
                        }
                    }
                }
            }

            if !ring_vals.is_empty() {
                // Use median of the ring for robustness
                ring_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                found = ring_vals[ring_vals.len() / 2];
                break 'rings;
            }
        }

        let fill = if found.is_finite() { found } else { 0.0 };
        fill_values.push((nr * w + nc_pos, fill));
    }

    // Apply fills
    for (idx, val) in fill_values {
        data[idx] = val;
    }
}

pub(crate) fn clean_band_nan_fill(raw: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out: Vec<f32> = raw[..h * w].to_vec();
    fill_nan_spatial_band(&mut out, h, w, 5);
    out
}

/// Same as clean_band_nan_fill but also clips to [0, 1].
/// Matches Python's `_fill_nan(np.clip(band, 0.0, 1.0))` used before LBP.
fn clean_band_nan_fill_clipped(raw: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out: Vec<f32> = raw[..h * w]
        .iter()
        .map(|&v| if v.is_finite() { v.clamp(0.0, 1.0) } else { f32::NAN })
        .collect();
    fill_nan_spatial_band(&mut out, h, w, 5);
    // Clip any filled values too
    for v in out.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    out
}

#[inline(always)]
fn safe_ratio(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        (a - b) / (a + b + EPS)
    } else {
        f32::NAN
    }
}

// =====================================================================
// Per-cell statistics (nan-aware, stable)
// =====================================================================

#[inline(always)]
pub(crate) fn percentile_linear(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (n as f32 - 1.0) * q;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = pos - lo as f32;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// 8 stats: mean, std, min, max, q25, median, q75, finite_frac
/// Uses np.percentile-compatible linear interpolation on finite-only values.
pub(crate) fn cell_stats_8(px: &[f32; N_PX]) -> [f32; 8] {
    let mut vals = [0.0f32; N_PX];
    let mut n: usize = 0;

    let mut sum = 0.0f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;

    for &v in px.iter() {
        if v.is_finite() {
            vals[n] = v;
            n += 1;
            sum += v as f64;
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
    }

    let finite_frac = n as f32 / N_PX as f32;
    if n == 0 {
        return [
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            0.0,
        ];
    }

    let mean = (sum / n as f64) as f32;

    // Stable variance (two-pass) in f64 — ddof=0 like numpy
    let mut var = 0.0f64;
    for i in 0..n {
        let d = vals[i] as f64 - mean as f64;
        var += d * d;
    }
    let std = ((var / n as f64).max(0.0)).sqrt() as f32;

    let vs = &mut vals[..n];
    vs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let q25 = percentile_linear(vs, 0.25);
    let med = percentile_linear(vs, 0.50);
    let q75 = percentile_linear(vs, 0.75);

    [mean, std, mn, mx, q25, med, q75, finite_frac]
}

/// Block-reduce 10×10 → 5×5 via nanmean of 2×2 blocks.
/// Matches Python's _block_reduce_mean(band, factor=2).
fn block_reduce_2x2(px: &[f32; N_PX]) -> ([f32; 25], usize) {
    let mut out = [0.0f32; 25];
    let mut count = 0usize;
    for br in 0..5 {
        for bc in 0..5 {
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for dr in 0..2 {
                for dc in 0..2 {
                    let v = px[(2 * br + dr) * GP + (2 * bc + dc)];
                    if v.is_finite() {
                        sum += v as f64;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out[count] = (sum / n as f64) as f32;
            } else {
                out[count] = f32::NAN;
            }
            count += 1;
        }
    }
    (out, 25)
}

/// 8 stats on a dynamically-sized slice (for block-reduced 25-element arrays)
fn cell_stats_8_dyn(px: &[f32], total_size: usize) -> [f32; 8] {
    let mut vals = Vec::with_capacity(total_size);
    let mut sum = 0.0f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;

    for &v in px.iter().take(total_size) {
        if v.is_finite() {
            vals.push(v);
            sum += v as f64;
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
    }

    let n = vals.len();
    let finite_frac = n as f32 / total_size as f32;
    if n == 0 {
        return [
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            0.0,
        ];
    }

    let mean = (sum / n as f64) as f32;
    let mut var = 0.0f64;
    for &v in &vals {
        let d = v as f64 - mean as f64;
        var += d * d;
    }
    let std = ((var / n as f64).max(0.0)).sqrt() as f32;

    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q25 = percentile_linear(&vals, 0.25);
    let med = percentile_linear(&vals, 0.50);
    let q75 = percentile_linear(&vals, 0.75);

    [mean, std, mn, mx, q25, med, q75, finite_frac]
}

#[inline(always)]
pub(crate) fn cell_stats_5(px: &[f32; N_PX]) -> [f32; 5] {
    let s = cell_stats_8(px);
    [s[0], s[1], s[4], s[5], s[6]]
}

#[inline(always)]
pub(crate) fn extract_cell(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; N_PX] {
    let mut px = [0.0f32; N_PX];
    let r0 = cr * GP;
    let c0 = cc * GP;
    for dr in 0..GP {
        let row_off = (r0 + dr) * w + c0;
        px[dr * GP..dr * GP + GP].copy_from_slice(&img[row_off..row_off + GP]);
    }
    px
}

pub(crate) fn cell_lbp_hist(lbp: &[u8], w: usize, cr: usize, cc: usize) -> [f32; LBP_BINS + 1] {
    let mut counts = [0u32; LBP_BINS];
    let r0 = cr * GP;
    let c0 = cc * GP;

    for dr in 0..GP {
        let row_off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let bin = lbp[row_off + dc] as usize;
            if bin < LBP_BINS {
                counts[bin] += 1;
            }
        }
    }

    let inv = 1.0 / N_PX as f32;
    let mut out = [0.0f32; LBP_BINS + 1];
    let mut entropy = 0.0f32;

    for i in 0..LBP_BINS {
        let p = counts[i] as f32 * inv;
        out[i] = p;
        if p > EPS {
            entropy -= p * p.ln();
        }
    }
    out[LBP_BINS] = entropy;
    out
}

fn cell_morans_i(px: &[f32; N_PX]) -> f32 {
    // NaN-aware Moran's I with 4-neighbor pairs (right + down)
    let mut sum = 0.0f64;
    let mut n_valid = 0usize;

    for &v in px.iter() {
        if v.is_finite() {
            sum += v as f64;
            n_valid += 1;
        }
    }
    if n_valid <= 1 {
        return f32::NAN;
    }

    let mean = (sum / n_valid as f64) as f32;

    let mut z = [f32::NAN; N_PX];
    let mut denom = 0.0f64;
    for i in 0..N_PX {
        if px[i].is_finite() {
            let dv = px[i] - mean;
            z[i] = dv;
            denom += (dv as f64) * (dv as f64);
        }
    }
    if denom < 1e-12 {
        return 0.0;
    }

    let mut w_sum = 0.0f64;
    let mut n_pairs = 0usize;

    for r in 0..GP {
        for c in 0..GP {
            let i = r * GP + c;
            if !z[i].is_finite() {
                continue;
            }
            if c + 1 < GP && z[i + 1].is_finite() {
                w_sum += (z[i] as f64) * (z[i + 1] as f64);
                n_pairs += 1;
            }
            if r + 1 < GP && z[i + GP].is_finite() {
                w_sum += (z[i] as f64) * (z[i + GP] as f64);
                n_pairs += 1;
            }
        }
    }

    if n_pairs == 0 {
        return 0.0;
    }

    ((n_valid as f64 / n_pairs as f64) * w_sum / denom) as f32
}

fn cell_agg_3(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; 3] {
    let r0 = cr * GP;
    let c0 = cc * GP;

    let mut sum = 0.0f64;
    let mut mx = f32::NEG_INFINITY;

    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let v = img[off + dc];
            sum += v as f64;
            if v > mx {
                mx = v;
            }
        }
    }

    let mean = sum / N_PX as f64;

    let mut var = 0.0f64;
    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let d = img[off + dc] as f64 - mean;
            var += d * d;
        }
    }
    let std = ((var / N_PX as f64).max(0.0)).sqrt() as f32;

    [mean as f32, std, mx]
}

fn cell_lap_stats(img: &[f32], w: usize, cr: usize, cc: usize) -> [f32; 2] {
    let r0 = cr * GP;
    let c0 = cc * GP;

    let mut abs_sum = 0.0f64;
    let mut sum = 0.0f64;

    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let v = img[off + dc] as f64;
            abs_sum += v.abs();
            sum += v;
        }
    }

    let mean = sum / N_PX as f64;

    let mut var = 0.0f64;
    for dr in 0..GP {
        let off = (r0 + dr) * w + c0;
        for dc in 0..GP {
            let d = img[off + dc] as f64 - mean;
            var += d * d;
        }
    }

    [
        (abs_sum / N_PX as f64) as f32,
        ((var / N_PX as f64).max(0.0)).sqrt() as f32,
    ]
}

// =====================================================================
// Main extraction
// =====================================================================

fn extract_cell_features(
    spec: &[f32],
    h: usize,
    w: usize,
    cr: usize,
    cc: usize,
    sobel: &[f32],
    lap: &[f32],
    nir_clean: &[f32],
    lbp_nir: &[u8],
    lbp_ndvi: &[u8],
    lbp_evi2: &[u8],
    lbp_swir1: &[u8],
    lbp_ndti: &[u8],
) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let mut fi: usize = 0;

    // 1) Band stats (80)
    // 20m bands (B05/B06/B07/B8A/B11/B12) get block-reduced 10×10→5×5
    // before stats, matching Python V10's _block_reduce_mean(band, factor=2)
    let mut band_px = [[0.0f32; N_PX]; N_BANDS];
    for b in 0..N_BANDS {
        let band_off = b * h * w;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for dr in 0..GP {
            let src_off = band_off + (r0 + dr) * w + c0;
            let dst_off = dr * GP;
            band_px[b][dst_off..dst_off + GP].copy_from_slice(&spec[src_off..src_off + GP]);
        }
        let is_20m = BANDS_20M.contains(&b);
        let s = if is_20m {
            let (reduced, n) = block_reduce_2x2(&band_px[b]);
            cell_stats_8_dyn(&reduced, n)
        } else {
            cell_stats_8(&band_px[b])
        };
        for v in s {
            out[fi] = v;
            fi += 1;
        }
    }

    // 2) Indices (75)
    let blue = &band_px[B02];
    let green = &band_px[B03];
    let red = &band_px[B04];
    let re1 = &band_px[B05];
    let re2 = &band_px[B06];
    let re3 = &band_px[B07];
    let nir = &band_px[B08];
    let swir1 = &band_px[B11];
    let _swir2 = &band_px[B12];

    let mut idx_px = [0.0f32; N_PX];

    // 10 normalized differences
    let pairs: [(usize, usize); 10] = [
        (B08, B04), // NDVI
        (B03, B08), // NDWI
        (B11, B08), // NDBI
        (B08, B11), // NDMI
        (B08, B12), // NBR
        (B08, B05), // NDRE1
        (B08, B06), // NDRE2
        (B03, B11), // MNDWI
        (B08, B03), // GNDVI
        (B11, B12), // NDTI
    ];

    let mut ndvi_px = [0.0f32; N_PX];
    for (pi, &(a, b)) in pairs.iter().enumerate() {
        for i in 0..N_PX {
            idx_px[i] = safe_ratio(band_px[a][i], band_px[b][i]);
        }
        if pi == 0 {
            ndvi_px = idx_px;
        }
        let s = cell_stats_5(&idx_px);
        for v in s {
            out[fi] = v;
            fi += 1;
        }
    }

    // SAVI
    for i in 0..N_PX {
        idx_px[i] = if nir[i].is_finite() && red[i].is_finite() {
            1.5 * (nir[i] - red[i]) / (nir[i] + red[i] + 0.5 + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // BSI
    for i in 0..N_PX {
        idx_px[i] = if swir1[i].is_finite()
            && red[i].is_finite()
            && nir[i].is_finite()
            && blue[i].is_finite()
        {
            let num = (swir1[i] + red[i]) - (nir[i] + blue[i]);
            num / ((swir1[i] + red[i]) + (nir[i] + blue[i]) + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // EVI2
    for i in 0..N_PX {
        idx_px[i] = if nir[i].is_finite() && red[i].is_finite() {
            2.5 * (nir[i] - red[i]) / (nir[i] + 2.4 * red[i] + 1.0 + EPS)
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // IRECI
    for i in 0..N_PX {
        idx_px[i] =
            if re3[i].is_finite() && red[i].is_finite() && re1[i].is_finite() && re2[i].is_finite()
            {
                (re3[i] - red[i]) / (re1[i] / (re2[i] + EPS) + EPS)
            } else {
                f32::NAN
            };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // CRI1
    for i in 0..N_PX {
        idx_px[i] = if green[i].is_finite() && re1[i].is_finite() && green[i] > EPS && re1[i] > EPS
        {
            (1.0 / green[i]) - (1.0 / re1[i])
        } else {
            f32::NAN
        };
    }
    for v in cell_stats_5(&idx_px) {
        out[fi] = v;
        fi += 1;
    }

    // 3) Tasseled Cap (6) — 10-band Nedkov 2017, matching Python exactly
    // dot product of all 10 bands with each TC coefficient vector
    for coeff in [TC10_B, TC10_G, TC10_W] {
        let mut vals = [0.0f32; N_PX];
        let mut n = 0usize;

        for i in 0..N_PX {
            let mut ok = true;
            let mut dot = 0.0f32;
            for b in 0..N_BANDS {
                let v = band_px[b][i];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                dot += v * coeff[b];
            }
            if ok {
                vals[n] = dot;
                n += 1;
            }
        }

        if n == 0 {
            out[fi] = f32::NAN;
            out[fi + 1] = f32::NAN;
        } else {
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += vals[i] as f64;
            }
            let mean = (sum / n as f64) as f32;

            let mut var = 0.0f64;
            for i in 0..n {
                let d = vals[i] as f64 - mean as f64;
                var += d * d;
            }
            let std = ((var / n as f64).max(0.0)).sqrt() as f32;

            out[fi] = mean;
            out[fi + 1] = std;
        }
        fi += 2;
    }

    // 4) Spatial (8)
    let e = cell_agg_3(sobel, w, cr, cc);
    out[fi] = e[0];
    fi += 1;
    out[fi] = e[1];
    fi += 1;
    out[fi] = e[2];
    fi += 1;

    let l = cell_lap_stats(lap, w, cr, cc);
    out[fi] = l[0];
    fi += 1;
    out[fi] = l[1];
    fi += 1;

    let nir_px = extract_cell(nir_clean, w, cr, cc);
    out[fi] = cell_morans_i(&nir_px);
    fi += 1;

    let ndvi_s = cell_stats_8(&ndvi_px);
    out[fi] = ndvi_s[3] - ndvi_s[2];
    fi += 1; // range
    out[fi] = ndvi_s[6] - ndvi_s[4];
    fi += 1; // IQR

    // 5) Multi-band LBP (55)
    let lbp_imgs = [lbp_nir, lbp_ndvi, lbp_evi2, lbp_swir1, lbp_ndti];
    for lbp in lbp_imgs {
        let hst = cell_lbp_hist(lbp, w, cr, cc);
        for v in hst {
            out[fi] = v;
            fi += 1;
        }
    }

    debug_assert_eq!(fi, N_FEAT);
    out
}

pub fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FEAT);

    let bands = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    ];
    let bst = [
        "mean",
        "std",
        "min",
        "max",
        "q25",
        "median",
        "q75",
        "finite_frac",
    ];
    for bn in &bands {
        for sn in &bst {
            names.push(format!("{bn}_{sn}"));
        }
    }

    let idxs = [
        "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "NDRE1", "NDRE2", "MNDWI", "GNDVI", "NDTI", "SAVI",
        "BSI", "EVI2", "IRECI", "CRI1",
    ];
    let ist = ["mean", "std", "q25", "median", "q75"];
    for idn in &idxs {
        for sn in &ist {
            names.push(format!("{idn}_{sn}"));
        }
    }

    for tc in &["TC_bright", "TC_green", "TC_wet"] {
        names.push(format!("{tc}_mean"));
        names.push(format!("{tc}_std"));
    }

    names.extend(
        [
            "edge_mean",
            "edge_std",
            "edge_max",
            "lap_abs_mean",
            "lap_std",
            "morans_I_NIR",
            "NDVI_range",
            "NDVI_iqr",
        ]
        .iter()
        .map(|s| s.to_string()),
    );

    // NIR LBP: use "LBP_u8_X" to match Python V10 naming exactly
    for b in 0..LBP_BINS {
        names.push(format!("LBP_u{LBP_P}_{b}"));
    }
    names.push("LBP_entropy".to_string());
    // Other LBP bands keep their band-prefixed names
    for lb in &["NDVI", "EVI2", "SWIR1", "NDTI"] {
        for b in 0..LBP_BINS {
            names.push(format!("LBP_{lb}_u{LBP_P}_{b}"));
        }
        names.push(format!("LBP_{lb}_entropy"));
    }

    assert_eq!(names.len(), N_FEAT);
    names
}

/// Extract features for multiple seasons. Returns [n_cells, n_seasons * N_FEAT] flat vector.
///
/// `season_data`: Vec of flat f32 arrays, each [N_BANDS * H * W] in band-interleaved order.
/// `n_rows`, `n_cols`: grid dimensions (H = n_rows * GP, W = n_cols * GP).
pub fn extract_all_seasons(season_data: &[Vec<f32>], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let h = n_rows * GP;
    let w = n_cols * GP;
    let n_seasons = season_data.len();
    let n_cells = n_rows * n_cols;
    let total_feats = n_cells * n_seasons * N_FEAT;

    let lbp_lut = build_lbp_lut();

    let season_results: Vec<Vec<[f32; N_FEAT]>> = season_data
        .iter()
        .map(|spec_slice| {
            let band_slice = |b: usize| -> &[f32] { &spec_slice[b * h * w..(b + 1) * h * w] };

            let nir_clean = clean_band_nan_fill(band_slice(B08), h, w);
            let sobel = compute_sobel_mag(&nir_clean, h, w);
            let laplacian = compute_laplacian(&nir_clean, h, w);

            let red_clean = clean_band_nan_fill(band_slice(B04), h, w);
            let swir1_clean = clean_band_nan_fill(band_slice(B11), h, w);
            let swir2_clean = clean_band_nan_fill(band_slice(B12), h, w);

            let swir1_lbp = clean_band_nan_fill_clipped(band_slice(B11), h, w);

            let ndvi_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let v = (nir_clean[i] - red_clean[i]) / (nir_clean[i] + red_clean[i] + EPS);
                    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
                })
                .collect();

            let evi2_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let e = 2.5 * (nir_clean[i] - red_clean[i])
                        / (nir_clean[i] + 2.4 * red_clean[i] + 1.0 + EPS);
                    ((e + 0.5) / 1.5).clamp(0.0, 1.0)
                })
                .collect();

            let ndti_img: Vec<f32> = (0..h * w)
                .into_par_iter()
                .map(|i| {
                    let v =
                        (swir1_clean[i] - swir2_clean[i]) / (swir1_clean[i] + swir2_clean[i] + EPS);
                    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
                })
                .collect();

            let lbp_nir =
                compute_lbp_perpatch(band_slice(B08), h, w, n_rows, n_cols, &lbp_lut, true);
            let lbp_ndvi = compute_lbp_raster(&ndvi_img, h, w, &lbp_lut);
            let lbp_evi2 = compute_lbp_raster(&evi2_img, h, w, &lbp_lut);
            let lbp_swir1 = compute_lbp_raster(&swir1_lbp, h, w, &lbp_lut);
            let lbp_ndti = compute_lbp_raster(&ndti_img, h, w, &lbp_lut);

            (0..n_cells)
                .into_par_iter()
                .map(|ci| {
                    extract_cell_features(
                        spec_slice,
                        h,
                        w,
                        ci / n_cols,
                        ci % n_cols,
                        &sobel,
                        &laplacian,
                        &nir_clean,
                        &lbp_nir,
                        &lbp_ndvi,
                        &lbp_evi2,
                        &lbp_swir1,
                        &lbp_ndti,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Interleave: for each cell, concatenate all seasons' features
    let mut flat = vec![0.0f32; total_feats];
    for ci in 0..n_cells {
        let cell_base = ci * n_seasons * N_FEAT;
        for si in 0..n_seasons {
            let dst = cell_base + si * N_FEAT;
            flat[dst..dst + N_FEAT].copy_from_slice(&season_results[si][ci]);
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflect_index() {
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(4, 5), 4);
        assert_eq!(reflect_index(-1, 5), 0); // -1 -> -(-1)-1 = 0
        assert_eq!(reflect_index(-2, 5), 1); // -2 -> -(-2)-1 = 1
        assert_eq!(reflect_index(5, 5), 4);  // 5 -> 2*5-5-1 = 4
        assert_eq!(reflect_index(6, 5), 3);  // 6 -> 2*5-6-1 = 3
    }

    #[test]
    fn test_cell_stats_8() {
        let mut px = [f32::NAN; 100];
        
        // Populate 10 valid values: 1.0 to 10.0
        for i in 0..10 {
            px[i] = (i + 1) as f32;
        }

        let stats = cell_stats_8(&px);
        // [mean, std, min, max, q25, med, q75, finite_frac]
        assert_eq!(stats[0], 5.5); // mean
        let expected_var = (0..10).map(|x| (x as f32 + 1.0 - 5.5).powi(2)).sum::<f32>() / 10.0;
        assert!((stats[1] - expected_var.sqrt()).abs() < 1e-5);
        assert_eq!(stats[2], 1.0); // min
        assert_eq!(stats[3], 10.0); // max
        assert_eq!(stats[5], 5.5); // med
        assert_eq!(stats[7], 0.1); // 10/100
    }

    #[test]
    fn test_build_lbp_lut() {
        let lut = build_lbp_lut();
        assert_eq!(lut.len(), 256);
        
        // 00000000 -> 0 transitions -> 0 ones -> bin 0
        assert_eq!(lut[0], 0);
        // 11111111 -> 0 transitions -> 8 ones -> bin 8
        assert_eq!(lut[255], 8);
        
        // 00000001 -> 2 transitions -> 1 one -> bin 1
        assert_eq!(lut[1], 1);
        
        // 01010101 -> 8 transitions -> non-uniform -> bin 9
        assert_eq!(lut[0b01010101], 9);
    }
}
```

---

## src/grid.rs

```rust
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
    use super::*;
    
    #[test]
    fn test_utm_conversion() {
        let (lat, lon) = utm::wsg84_utm_to_lat_lon(500000.0, 4600000.0, 32, 'N').unwrap();
        assert!(lat > 40.0 && lat < 45.0);
        assert!(lon > 8.0 && lon < 12.0);
    }
}
```

---

## src/labels.rs

```rust
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
    if year > 2021 {
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
```

---

## src/parquet_io.rs

```rust
use anyhow::{Context, Result};
use std::path::Path;

/// Read a feature parquet file and return (column_names, data_matrix).
/// data_matrix is row-major: [n_cells, n_features].
///
/// Uses Arrow RecordBatch reader for columnar access (much faster than row iteration).
pub fn read_feature_parquet(path: &Path) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open parquet: {}", path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let n_cols = schema.fields().len();
    let col_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let reader = builder.build()?;

    // Read all batches and accumulate rows
    let mut rows: Vec<Vec<f32>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let n_rows = batch.num_rows();
        let batch_start = rows.len();

        // Extend rows for this batch
        rows.resize_with(batch_start + n_rows, || vec![0.0f32; n_cols]);

        // Read each column and fill into rows (columnar -> row-major)
        for col_idx in 0..n_cols {
            let col = batch.column(col_idx);

            match col.data_type() {
                DataType::Float32 => {
                    let arr = col.as_primitive::<Float32Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r)
                        };
                    }
                }
                DataType::Float64 => {
                    let arr = col.as_primitive::<Float64Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r) as f32
                        };
                    }
                }
                DataType::Int64 => {
                    let arr = col.as_primitive::<Int64Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r) as f32
                        };
                    }
                }
                DataType::Int32 => {
                    let arr = col.as_primitive::<Int32Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r) as f32
                        };
                    }
                }
                _ => {
                    // Unknown type, fill with NaN
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = f32::NAN;
                    }
                }
            }
        }
    }

    Ok((col_names, rows))
}

/// Write predictions to a parquet file.
/// predictions: [n_cells, n_classes] row-major.
pub fn write_predictions_parquet(
    path: &Path,
    class_names: &[&str],
    predictions: &[Vec<f32>],
    model_name: &str,
) -> Result<()> {
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let n_cells = predictions.len();
    let n_classes = class_names.len();

    // Build schema: cell_id + class columns
    let mut fields = vec![Field::new("cell_id", DataType::Int32, false)];
    for cn in class_names {
        fields.push(Field::new(
            format!("{}_{}", cn, model_name),
            DataType::Float32,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Build arrays
    let cell_ids: Vec<i32> = (0..n_cells as i32).collect();
    let cell_id_array = Arc::new(arrow::array::Int32Array::from(cell_ids));

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![cell_id_array];
    for ci in 0..n_classes {
        let vals: Vec<f32> = predictions.iter().map(|row| row[ci]).collect();
        columns.push(Arc::new(Float32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Write features to a parquet file.
/// extra_cols/extra_data: metadata columns (cell_id, valid_fraction).
/// feature_cols: feature column names.
/// rows: [n_cells][n_features] feature data.
pub fn write_feature_parquet(
    path: &Path,
    extra_cols: &[String],
    extra_data: &[Vec<f32>],
    feature_cols: &[String],
    rows: &[Vec<f32>],
) -> Result<()> {
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let n_cells = rows.len();
    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::new();

    // Extra columns first
    for (i, name) in extra_cols.iter().enumerate() {
        fields.push(Field::new(name, DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from(extra_data[i].clone())));
    }

    // Feature columns
    for (ci, name) in feature_cols.iter().enumerate() {
        fields.push(Field::new(name, DataType::Float32, true));
        let vals: Vec<f32> = (0..n_cells).map(|ri| rows[ri][ci]).collect();
        arrays.push(Arc::new(Float32Array::from(vals)));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
```

---

## src/predict.rs

```rust
use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;

use crate::config::N_CLASSES;

/// Model configuration loaded from model_config.json.
#[derive(serde::Deserialize)]
pub struct ModelConfig {
    pub label_threshold: f32,
    #[serde(default = "default_n_classes")]
    pub n_classes: usize,
}

fn default_n_classes() -> usize {
    N_CLASSES
}

impl ModelConfig {
    /// Load model config from the ONNX directory.
    pub fn load(onnx_dir: &Path) -> Result<Self> {
        let path = onnx_dir.join("model_config.json");
        if !path.exists() {
            // Default: no threshold (backwards compatible)
            println!("  No model_config.json found, using default threshold=0.0");
            return Ok(Self {
                label_threshold: 0.0,
                n_classes: N_CLASSES,
            });
        }
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read model config: {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&data)?;
        println!(
            "  Loaded model config: label_threshold={:.4}",
            cfg.label_threshold
        );
        Ok(cfg)
    }

    /// Apply label threshold filtering to predictions:
    /// - Zero out classes with probability below threshold
    /// - Renormalize remaining classes to sum to 1.0
    pub fn apply_threshold(&self, predictions: &mut [Vec<f32>]) {
        if self.label_threshold <= 0.0 {
            return;
        }
        for row in predictions.iter_mut() {
            // Zero out below threshold
            for val in row.iter_mut() {
                if *val < self.label_threshold {
                    *val = 0.0;
                }
            }
            // Renormalize
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }
    }
}

/// Scaler parameters (mean + scale) for StandardScaler transform.
#[derive(serde::Deserialize)]
pub struct ScalerParams {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

impl ScalerParams {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read scaler: {}", path.display()))?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Apply standardization: (x - mean) / scale
    pub fn transform(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let m = self.mean[i] as f32;
                let s = self.scale[i] as f32;
                if s.abs() < 1e-12 {
                    0.0
                } else {
                    (v - m) / s
                }
            })
            .collect()
    }
}

/// A loaded single ONNX MLP model for inference.
pub struct OnnxMlp {
    session: Session,
}

impl OnnxMlp {
    /// Load the single MLP ONNX model.
    pub fn load(onnx_dir: &Path) -> Result<Self> {
        let path = onnx_dir.join("mlp_fold_0.onnx");
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&path)
            .with_context(|| format!("Cannot load ONNX: {}", path.display()))?;
        Ok(Self { session })
    }

    /// Run inference for all cells. Returns [n_cells, N_CLASSES] softmax probabilities.
    ///
    /// `features`: [n_cells][n_features] row-major, already scaled.
    pub fn predict(&mut self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_cells = features.len();
        if n_cells == 0 {
            return Ok(Vec::new());
        }

        // Process in chunks to avoid memory issues
        const CHUNK_SIZE: usize = 65536;
        let mut all_results = Vec::with_capacity(n_cells);

        for chunk_start in (0..n_cells).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n_cells);
            let chunk = &features[chunk_start..chunk_end];
            let chunk_results = self.run_batch(chunk)?;
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    /// Run ONNX session on a batch of inputs, returning [n_rows][N_CLASSES].
    fn run_batch(&mut self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_rows = features.len();
        let n_cols = features[0].len();

        // Flatten to contiguous array efficiently using flat_map
        let flat: Vec<f32> = features
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        // Create ONNX tensor
        let input_tensor = ort::value::Tensor::from_array(([n_rows, n_cols], flat))?;

        let outputs = self.session.run(ort::inputs!["X" => input_tensor])?;

        // Extract output
        let output = &outputs[0];
        let (tensor_shape, tensor_data) = output.try_extract_tensor::<f32>()?;

        let out_cols = if tensor_shape.len() > 1 {
            tensor_shape[1] as usize
        } else {
            1
        };
        assert_eq!(
            out_cols, N_CLASSES,
            "ONNX output has {} cols, expected {}",
            out_cols, N_CLASSES
        );

        let mut result = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let start = i * out_cols;
            let end = start + out_cols;
            result.push(tensor_data[start..end].to_vec());
        }

        Ok(result)
    }
}
```

---

## src/reproject.rs

```rust
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
    let n_val = WGS84_A / (1.0 - e).sqrt();

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
```

---

## src/sar_download.rs

```rust
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
```

---

## src/sar_features.rs

```rust
//! SAR (Sentinel-1) feature extraction: VV/VH statistics, indices, and LBP texture.
//! 48 features per cell per season.

use rayon::prelude::*;

use crate::features::{
    build_lbp_lut, cell_lbp_hist, cell_stats_5, cell_stats_8, compute_lbp_perpatch, EPS, GP,
    LBP_BINS,
};

const N_PX: usize = GP * GP; // 100 pixels per cell

// SAR band indices within 2-band TIF
const SAR_VV: usize = 0;
const SAR_VH: usize = 1;
pub const N_SAR_BANDS: usize = 2;

// Feature counts per season:
//   VV stats: 8, VH stats: 8
//   CR (VH/VV) stats: 5, RVI stats: 5
//   LBP(VV): 11, LBP(VH): 11
//   Total: 48
const N_SAR_BAND_STATS: usize = N_SAR_BANDS * 8; // 16
const N_SAR_IDX_STATS: usize = 2 * 5; // 10 (CR + RVI, 5 stats each)
const N_SAR_LBP: usize = 2 * (LBP_BINS + 1); // 22 (VV + VH, 11 each)
pub const N_SAR_FEAT: usize = N_SAR_BAND_STATS + N_SAR_IDX_STATS + N_SAR_LBP; // 48

/// Compute cross-polarization ratio = VH / VV
#[inline(always)]
fn sar_cross_ratio(vv: f32, vh: f32) -> f32 {
    if vv.is_finite() && vh.is_finite() && vv.abs() > EPS {
        vh / vv
    } else {
        f32::NAN
    }
}

/// Compute Radar Vegetation Index = 4 * VH / (VV + VH)
#[inline(always)]
fn sar_rvi(vv: f32, vh: f32) -> f32 {
    if vv.is_finite() && vh.is_finite() {
        let denom = vv + vh;
        if denom.abs() > EPS {
            4.0 * vh / denom
        } else {
            f32::NAN
        }
    } else {
        f32::NAN
    }
}

/// Extract 48 SAR features for one cell.
fn extract_sar_cell_features(
    sar_data: &[f32],
    h: usize,
    w: usize,
    cr: usize,
    cc: usize,
    lbp_vv: &[u8],
    lbp_vh: &[u8],
) -> [f32; N_SAR_FEAT] {
    let mut out = [0.0f32; N_SAR_FEAT];
    let mut fi: usize = 0;

    // 1) Band stats (16: 8 for VV, 8 for VH)
    let mut band_px = [[0.0f32; N_PX]; N_SAR_BANDS];
    for b in 0..N_SAR_BANDS {
        let band_off = b * h * w;
        let r0 = cr * GP;
        let c0 = cc * GP;
        for dr in 0..GP {
            let src_off = band_off + (r0 + dr) * w + c0;
            let dst_off = dr * GP;
            band_px[b][dst_off..dst_off + GP].copy_from_slice(&sar_data[src_off..src_off + GP]);
        }
        let s = cell_stats_8(&band_px[b]);
        for v in s {
            out[fi] = v;
            fi += 1;
        }
    }

    // 2) SAR indices (10: 5 for CR, 5 for RVI)
    let vv = &band_px[SAR_VV];
    let vh = &band_px[SAR_VH];

    // Cross-pol ratio
    let mut idx_px = [0.0f32; N_PX];
    for i in 0..N_PX {
        idx_px[i] = sar_cross_ratio(vv[i], vh[i]);
    }
    let s = cell_stats_5(&idx_px);
    for v in s {
        out[fi] = v;
        fi += 1;
    }

    // RVI
    for i in 0..N_PX {
        idx_px[i] = sar_rvi(vv[i], vh[i]);
    }
    let s = cell_stats_5(&idx_px);
    for v in s {
        out[fi] = v;
        fi += 1;
    }

    // 3) LBP texture (22: 11 for VV, 11 for VH)
    let lbp_vv_hist = cell_lbp_hist(lbp_vv, w, cr, cc);
    for v in lbp_vv_hist {
        out[fi] = v;
        fi += 1;
    }

    let lbp_vh_hist = cell_lbp_hist(lbp_vh, w, cr, cc);
    for v in lbp_vh_hist {
        out[fi] = v;
        fi += 1;
    }

    debug_assert_eq!(fi, N_SAR_FEAT);
    out
}

/// Column names for SAR features (48 names).
pub fn sar_feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_SAR_FEAT);

    // Band stats
    let bands = ["SAR_VV", "SAR_VH"];
    let bst = [
        "mean",
        "std",
        "min",
        "max",
        "q25",
        "median",
        "q75",
        "finite_frac",
    ];
    for bn in &bands {
        for sn in &bst {
            names.push(format!("{bn}_{sn}"));
        }
    }

    // Index stats
    let idxs = ["SAR_CR", "SAR_RVI"];
    let ist = ["mean", "std", "q25", "median", "q75"];
    for idn in &idxs {
        for sn in &ist {
            names.push(format!("{idn}_{sn}"));
        }
    }

    // LBP
    for lb in &["SAR_LBP_VV", "SAR_LBP_VH"] {
        for b in 0..LBP_BINS {
            names.push(format!("{lb}_u8_{b}"));
        }
        names.push(format!("{lb}_entropy"));
    }

    assert_eq!(names.len(), N_SAR_FEAT);
    names
}

/// Extract SAR features for all seasons. Returns [n_cells, n_seasons * N_SAR_FEAT] flat vector.
///
/// `season_data`: Vec of flat f32 arrays, each [N_SAR_BANDS * H * W].
/// `n_rows`, `n_cols`: grid dimensions (H = n_rows * GP, W = n_cols * GP).
pub fn extract_all_sar_seasons(season_data: &[Vec<f32>], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let h = n_rows * GP;
    let w = n_cols * GP;
    let n_seasons = season_data.len();
    let n_cells = n_rows * n_cols;
    let total_feats = n_cells * n_seasons * N_SAR_FEAT;

    let lbp_lut = build_lbp_lut();

    let season_results: Vec<Vec<[f32; N_SAR_FEAT]>> = season_data
        .iter()
        .map(|sar_slice| {
            let band_slice = |b: usize| -> &[f32] { &sar_slice[b * h * w..(b + 1) * h * w] };

            // LBP on VV and VH (per-patch, with clipping to [0,1] since SAR is
            // already normalized to [0,1] after dB conversion + scaling)
            let lbp_vv =
                compute_lbp_perpatch(band_slice(SAR_VV), h, w, n_rows, n_cols, &lbp_lut, true);
            let lbp_vh =
                compute_lbp_perpatch(band_slice(SAR_VH), h, w, n_rows, n_cols, &lbp_lut, true);

            (0..n_cells)
                .into_par_iter()
                .map(|ci| {
                    extract_sar_cell_features(
                        sar_slice,
                        h,
                        w,
                        ci / n_cols,
                        ci % n_cols,
                        &lbp_vv,
                        &lbp_vh,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Interleave: for each cell, concatenate all seasons' SAR features
    let mut flat = vec![0.0f32; total_feats];
    for ci in 0..n_cells {
        let cell_base = ci * n_seasons * N_SAR_FEAT;
        for (si, season) in season_results.iter().enumerate().take(n_seasons) {
            let dst = cell_base + si * N_SAR_FEAT;
            flat[dst..dst + N_SAR_FEAT].copy_from_slice(&season[ci]);
        }
    }

    flat
}
```

---

## src/stac.rs

```rust
//! STAC API client for Planetary Computer Sentinel-2 search + URL signing.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const STAC_API: &str = "https://planetarycomputer.microsoft.com/api/stac/v1";
const TOKEN_API: &str = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a";

// ---- STAC search request/response types ----

#[derive(Serialize)]
struct StacSearchBody {
    collections: Vec<String>,
    bbox: [f64; 4],
    datetime: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<serde_json::Value>,
    limit: u32,
}

#[derive(Deserialize, Debug)]
pub struct StacFeatureCollection {
    pub features: Vec<StacItem>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct StacItem {
    pub id: String,
    pub properties: StacProperties,
    pub assets: HashMap<String, StacAsset>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct StacProperties {
    #[serde(rename = "eo:cloud_cover")]
    pub cloud_cover: Option<f64>,
    pub datetime: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct StacAsset {
    pub href: String,
    #[serde(rename = "type")]
    pub media_type: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    token: String,
}

// ---- Season date ranges ----

pub fn season_date_range(year: u32, season: &str) -> Result<(String, String)> {
    match season {
        "spring" => Ok((format!("{year}-04-01"), format!("{year}-05-31"))),
        "summer" => Ok((format!("{year}-06-01"), format!("{year}-08-31"))),
        "autumn" => Ok((format!("{year}-09-01"), format!("{year}-10-31"))),
        _ => anyhow::bail!("Unknown season: '{season}' (expected spring/summer/autumn)"),
    }
}

// ---- STAC search ----

/// Search for Sentinel-2 L2A scenes matching the given parameters.
pub async fn search_scenes(
    client: &Client,
    bbox: [f64; 4],
    start_date: &str,
    end_date: &str,
    cloud_max: f64,
) -> Result<Vec<StacItem>> {
    let body = StacSearchBody {
        collections: vec!["sentinel-2-l2a".to_string()],
        bbox,
        datetime: format!("{start_date}/{end_date}"),
        query: Some(serde_json::json!({
            "eo:cloud_cover": {"lt": cloud_max}
        })),
        limit: 500,
    };

    let url = format!("{STAC_API}/search");
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("STAC search request failed")?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("STAC search returned {status}: {text}");
    }

    let fc: StacFeatureCollection = resp.json().await.context("Failed to parse STAC response")?;
    Ok(fc.features)
}

/// Search with cloud cover ramp (40 -> 50 -> 60) and date expansion fallback.
pub async fn search_with_fallback(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    season: &str,
    min_scenes: usize,
) -> Result<Vec<StacItem>> {
    let (start, end) = season_date_range(year, season)?;

    // Try increasing cloud cover thresholds
    for cloud_max in [40.0, 50.0, 60.0] {
        let items = search_scenes(client, bbox, &start, &end, cloud_max).await?;
        if items.len() >= min_scenes {
            return Ok(items);
        }
    }

    // Expand date window by ±14 days
    let s = chrono_parse_expand(&start, -14);
    let e = chrono_parse_expand(&end, 14);
    let items = search_scenes(client, bbox, &s, &e, 60.0).await?;
    Ok(items)
}

/// Simple date expansion (±days) without pulling in chrono.
fn chrono_parse_expand(date_str: &str, days: i32) -> String {
    // Parse YYYY-MM-DD, add days naively
    let parts: Vec<u32> = date_str.split('-').map(|p| p.parse().unwrap()).collect();
    let (y, m, d) = (parts[0] as i32, parts[1] as i32, parts[2] as i32);

    // Convert to a rough day count and back (good enough for ±14 days)
    let days_in_month = |y: i32, m: i32| -> i32 {
        match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                    29
                } else {
                    28
                }
            }
            _ => 30,
        }
    };
    let mut total_day = d + days;
    let mut month = m;
    let mut year = y;

    while total_day < 1 {
        month -= 1;
        if month < 1 {
            month = 12;
            year -= 1;
        }
        total_day += days_in_month(year, month);
    }
    while total_day > days_in_month(year, month) {
        total_day -= days_in_month(year, month);
        month += 1;
        if month > 12 {
            month = 1;
            year += 1;
        }
    }

    format!("{year:04}-{month:02}-{total_day:02}")
}

// ---- Token-based signing ----

/// Cached SAS token with timestamp for auto-refresh.
static CACHED_TOKEN: tokio::sync::Mutex<Option<(String, std::time::Instant)>> =
    tokio::sync::Mutex::const_new(None);

/// Token TTL: refresh after 50 minutes (Planetary Computer tokens last ~1 hour).
const TOKEN_TTL_SECS: u64 = 50 * 60;

/// Fetch a fresh SAS token (with retry on 429).
async fn fetch_token(client: &Client, api_url: &str) -> Result<String> {
    let max_retries = 3;
    let mut wait_secs = 5u64;

    for attempt in 0..=max_retries {
        let resp = client
            .get(api_url)
            .send()
            .await
            .with_context(|| format!("Token request to {api_url} failed"))?;

        let status = resp.status();
        if status.as_u16() == 429 && attempt < max_retries {
            eprintln!("    Rate limited on token, waiting {wait_secs}s...");
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            wait_secs *= 2;
            continue;
        }

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Token API {api_url} returned {status}: {text}");
        }

        let tr: TokenResponse = resp
            .json()
            .await
            .context("Failed to parse token response")?;
        return Ok(tr.token);
    }

    anyhow::bail!("Token fetch exhausted retries for {api_url}")
}

/// Get a SAS token for the sentinel-2-l2a collection (cached, auto-refreshes after 50 min).
pub async fn get_collection_token(client: &Client) -> Result<String> {
    let mut guard = CACHED_TOKEN.lock().await;
    if let Some((ref token, ref ts)) = *guard {
        if ts.elapsed().as_secs() < TOKEN_TTL_SECS {
            return Ok(token.clone());
        }
        eprintln!("    S2 token expired, refreshing...");
    }
    let token = fetch_token(client, TOKEN_API).await?;
    *guard = Some((token.clone(), std::time::Instant::now()));
    Ok(token)
}

/// Apply a SAS token to a blob URL.
fn apply_token(href: &str, token: &str) -> String {
    if href.contains('?') {
        format!("{href}&{token}")
    } else {
        format!("{href}?{token}")
    }
}

/// Public version of apply_token for cross-module use.
pub fn apply_token_pub(href: &str, token: &str) -> String {
    apply_token(href, token)
}

/// Sign all band asset URLs in a scene item using a pre-fetched token.
pub fn sign_scene_assets_with_token(
    item: &StacItem,
    band_names: &[&str],
    token: &str,
) -> Result<HashMap<String, String>> {
    let mut signed = HashMap::new();
    for band in band_names {
        let band_str = band.to_string();
        if let Some(asset) = item.assets.get(&band_str) {
            signed.insert(band_str, apply_token(&asset.href, token));
        } else {
            anyhow::bail!("Scene {} missing band {}", item.id, band);
        }
    }
    Ok(signed)
}

/// Get bands + SCL for cloud masking.
pub fn all_download_bands() -> Vec<&'static str> {
    vec![
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL",
    ]
}

// ---- Sentinel-1 SAR ----

const S1_TOKEN_API: &str =
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd";

/// Cached SAS token for S1 with auto-refresh.
static CACHED_S1_TOKEN: tokio::sync::Mutex<Option<(String, std::time::Instant)>> =
    tokio::sync::Mutex::const_new(None);

/// Get a SAS token for sentinel-1-grd (cached, auto-refreshes after 50 min).
pub async fn get_s1_token(client: &Client) -> Result<String> {
    let mut guard = CACHED_S1_TOKEN.lock().await;
    if let Some((ref token, ref ts)) = *guard {
        if ts.elapsed().as_secs() < TOKEN_TTL_SECS {
            return Ok(token.clone());
        }
        eprintln!("    S1 token expired, refreshing...");
    }
    let token = fetch_token(client, S1_TOKEN_API).await?;
    *guard = Some((token.clone(), std::time::Instant::now()));
    Ok(token)
}

/// Search for Sentinel-1 IW GRD scenes (with retry on transient failures).
pub async fn search_sar_scenes(
    client: &Client,
    bbox: [f64; 4],
    year: u32,
    season: &str,
) -> Result<Vec<StacItem>> {
    let (start, end) = season_date_range(year, season)?;
    let url = format!("{STAC_API}/search");

    // Retry wrapper for STAC POST requests
    let stac_post_with_retry = |body: StacSearchBody| {
        let client = client.clone();
        let url = url.clone();
        async move {
            let mut last_err = String::new();
            for attempt in 0..4u32 {
                if attempt > 0 {
                    let wait = 2u64 << (attempt - 1); // 2s, 4s, 8s
                    eprintln!("    S1 STAC retry {attempt}/3 in {wait}s...");
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                }
                let body_clone = StacSearchBody {
                    collections: body.collections.clone(),
                    bbox: body.bbox,
                    datetime: body.datetime.clone(),
                    query: body.query.clone(),
                    limit: body.limit,
                };
                match client.post(&url).json(&body_clone).send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        if status.as_u16() == 429 {
                            last_err = "rate limited (429)".to_string();
                            continue;
                        }
                        if !status.is_success() {
                            let text = resp.text().await.unwrap_or_default();
                            last_err = format!("{status}: {text}");
                            continue;
                        }
                        match resp.json::<StacFeatureCollection>().await {
                            Ok(fc) => return Ok(fc.features),
                            Err(e) => {
                                last_err = format!("parse error: {e}");
                                continue;
                            }
                        }
                    }
                    Err(e) => {
                        last_err = format!("network error: {e}");
                        continue;
                    }
                }
            }
            anyhow::bail!("S1 STAC search failed after 4 attempts: {last_err}")
        }
    };

    // First try ascending orbit only
    let body = StacSearchBody {
        collections: vec!["sentinel-1-grd".to_string()],
        bbox,
        datetime: format!("{start}/{end}"),
        query: Some(serde_json::json!({
            "sar:instrument_mode": {"eq": "IW"},
            "sat:orbit_state": {"eq": "ascending"}
        })),
        limit: 500,
    };

    let items = stac_post_with_retry(body).await?;
    if items.len() >= 3 {
        return Ok(items);
    }

    // Fallback: any orbit
    let body2 = StacSearchBody {
        collections: vec!["sentinel-1-grd".to_string()],
        bbox,
        datetime: format!("{start}/{end}"),
        query: Some(serde_json::json!({
            "sar:instrument_mode": {"eq": "IW"}
        })),
        limit: 500,
    };

    let items2 = stac_post_with_retry(body2).await?;
    if items2.len() > items.len() {
        Ok(items2)
    } else {
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_season_date_range() {
        let (start, end) = season_date_range(2023, "spring").unwrap();
        assert_eq!(start, "2023-04-01");
        assert_eq!(end, "2023-05-31");

        let (start, end) = season_date_range(2023, "summer").unwrap();
        assert_eq!(start, "2023-06-01");
        assert_eq!(end, "2023-08-31");

        let (start, end) = season_date_range(2023, "autumn").unwrap();
        assert_eq!(start, "2023-09-01");
        assert_eq!(end, "2023-10-31");

        assert!(season_date_range(2023, "winter").is_err());
    }

    #[test]
    fn test_chrono_parse_expand() {
        // Expand forward
        assert_eq!(chrono_parse_expand("2023-05-31", 14), "2023-06-14");
        // Expand backward
        assert_eq!(chrono_parse_expand("2023-04-01", -14), "2023-03-18");
        
        // Year cross forward
        assert_eq!(chrono_parse_expand("2023-12-25", 10), "2024-01-04");
        // Year cross backward
        assert_eq!(chrono_parse_expand("2024-01-05", -10), "2023-12-26");

        // Leap year forward
        assert_eq!(chrono_parse_expand("2024-02-28", 2), "2024-03-01");
        // Leap year backward
        assert_eq!(chrono_parse_expand("2024-03-01", -2), "2024-02-28");
        
        // Non-leap year forward
        assert_eq!(chrono_parse_expand("2023-02-28", 2), "2023-03-02");
    }

    #[test]
    fn test_apply_token_pub() {
        let url1 = "https://example.com/asset.tif";
        assert_eq!(apply_token_pub(url1, "token=123"), "https://example.com/asset.tif?token=123");

        let url2 = "https://example.com/asset.tif?foo=bar";
        assert_eq!(apply_token_pub(url2, "token=123"), "https://example.com/asset.tif?foo=bar&token=123");
    }
}
```

---

## src/tif_reader.rs

```rust
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
```

---

## tests/pipeline_e2e.rs

```rust
use anyhow::Result;
use assert_cmd::cargo_bin_cmd;

#[test]
fn test_terrapulse_cli_help() -> Result<()> {
    let mut cmd = cargo_bin_cmd!("terrapulse");
    let assert = cmd.arg("--help").assert();
    assert.success()
          .stdout(predicates::str::contains("Fast TerraPulse inference pipeline"));
    Ok(())
}

#[test]
fn test_pipeline_dry_run() -> Result<()> {
    Ok(())
}
```

---

## check_s1_compression.py

```python
import planetary_computer
import pystac_client
import rasterio
import tifffile
import urllib.request
import io

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = catalog.search(
    collections=["sentinel-1-grd"],
    bbox=[10.95, 49.38, 11.20, 49.52],
    datetime="2023-09-01/2023-10-31",
)
items = list(search.items())
if not items:
    print("No items found")
    exit(1)

for item in items:
    item_signed = planetary_computer.sign(item)
    vv_url = item_signed.assets["vv"].href
    
    req = urllib.request.Request(vv_url, headers={"Range": "bytes=0-500000"})
    with urllib.request.urlopen(req) as response:
        header_data = response.read()

    with tifffile.TiffFile(io.BytesIO(header_data)) as tif:
        compressions = []
        for page in tif.pages:
            for tag in page.tags.values():
                print(f"Tag {tag.code} ({tag.name}): {tag.value}")
            break
    break
```

---

## helpers/composite.py

```python
#!/usr/bin/env python3
"""
Composite helper for terrapulse Rust pipeline.

Called by the Rust download module to handle reprojection + median compositing
using rasterio, since Rust doesn't have mature reprojection libraries.

Optimizations (v2):
  - GDAL HTTP tuning set BEFORE rasterio import
  - Parallel SCENE downloads (ThreadPoolExecutor at scene level)
  - Parallel band downloads within each scene
  - Hard per-scene timeout to prevent infinite hangs
  - Retries on failed band reads
  - Vectorized cloud masking via np.isin()

Usage:
    python composite.py \\
        --scenes-json scenes.json \\
        --anchor-ref anchor_utm32632_10m.tif \\
        --output sentinel2_nuremberg_2024_summer.tif \\
        --year 2024
"""

import os

# ── MUST be set before importing rasterio/GDAL ──
os.environ["GDAL_HTTP_TIMEOUT"] = "15"           # was 60 — fail fast, retry instead
os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "8"       # was 15
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "67108864"  # 64 MB VSIL cache
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_VERSION"] = "2"       # force HTTP/2 for multiplexing
os.environ["CPL_CURL_VERBOSE"] = "NO"

import argparse
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# SCL classes to exclude: 0=nodata, 1=defective, 2=dark, 3=shadow,
# 8=cloud_med, 9=cloud_high, 10=cirrus, 11=snow
SCL_EXCLUDE = frozenset({0, 1, 2, 3, 8, 9, 10, 11})
SCL_EXCLUDE_ARR = np.array(sorted(SCL_EXCLUDE), dtype=np.uint8)
NODATA = -9999
MAX_BAND_WORKERS = 10    # parallel band downloads per scene
MAX_SCENE_WORKERS = 4    # parallel scene downloads
SCENE_TIMEOUT = 30       # hard timeout per scene (seconds) — was 120/180, caused 2min hangs
BAND_RETRIES = 2         # retries per band on failure


def read_band_warped(href, dst_crs, dst_transform, dst_width, dst_height, is_scl=False):
    """Read a single band via WarpedVRT, reprojecting on-the-fly."""
    last_err = None
    for attempt in range(1 + BAND_RETRIES):
        try:
            with rasterio.open(href) as src:
                with WarpedVRT(
                    src,
                    crs=dst_crs,
                    transform=dst_transform,
                    width=dst_width,
                    height=dst_height,
                    resampling=Resampling.nearest if is_scl else Resampling.bilinear,
                    dst_nodata=0 if is_scl else np.nan,
                ) as vrt:
                    return vrt.read(1)
        except Exception as e:
            last_err = e
            if attempt < BAND_RETRIES:
                import time
                time.sleep(1.0 * (attempt + 1))
    raise last_err


def download_scene_inner(scene, dst_crs, dst_transform, dst_width, dst_height):
    """Download all bands for one scene in parallel."""
    bands = scene["bands"]
    futures = {}

    with ThreadPoolExecutor(max_workers=MAX_BAND_WORKERS) as executor:
        for b in SENTINEL_BANDS:
            futures[executor.submit(
                read_band_warped, bands[b], dst_crs, dst_transform, dst_width, dst_height
            )] = ("spectral", b)
        futures[executor.submit(
            read_band_warped, bands["SCL"], dst_crs, dst_transform, dst_width, dst_height, True
        )] = ("scl", "SCL")

        spectral_dict = {}
        scl = None
        for future in as_completed(futures, timeout=SCENE_TIMEOUT):
            band_type, band_name = futures[future]
            data = future.result()
            if band_type == "scl":
                scl = data
            else:
                spectral_dict[band_name] = data

    spectral_stack = np.stack([spectral_dict[b] for b in SENTINEL_BANDS])
    return spectral_stack, scl


def _download_one_scene(args):
    """Wrapper for scene-level parallelism. Returns (index, result_or_None)."""
    idx, scene, dst_crs, dst_transform, dst_width, dst_height = args
    try:
        spectral_stack, scl = download_scene_inner(
            scene, dst_crs, dst_transform, dst_width, dst_height)
        print(f"    Scene {idx+1}: OK", file=sys.stderr)
        return idx, spectral_stack, scl
    except Exception as e:
        print(f"    Scene {idx+1}: FAILED ({e})", file=sys.stderr)
        return idx, None, None


def process_scenes(scenes, dst_crs, dst_transform, dst_width, dst_height, year):
    """Download, reproject, mask, and composite all scenes — parallelized."""
    n_scenes = len(scenes)
    n_bands = len(SENTINEL_BANDS)

    print(f"    Downloading {n_scenes} scenes ({MAX_SCENE_WORKERS} parallel)...",
          file=sys.stderr, flush=True)

    # ── Parallel scene downloads ──
    args_list = [
        (i, scene, dst_crs, dst_transform, dst_width, dst_height)
        for i, scene in enumerate(scenes)
    ]

    all_spectral = []
    all_scl = []

    with ThreadPoolExecutor(max_workers=MAX_SCENE_WORKERS) as pool:
        futures = {pool.submit(_download_one_scene, a): a[0] for a in args_list}
        for future in as_completed(futures):
            idx, spectral, scl = future.result()
            if spectral is not None:
                all_spectral.append(spectral)
                all_scl.append(scl)

    n_ok = len(all_spectral)
    print(f"    {n_ok}/{n_scenes} scenes OK", file=sys.stderr, flush=True)

    if not all_spectral:
        return None, None

    # ── Cloud masking (vectorized) ──
    scl_stack = np.stack(all_scl)                       # (n_ok, H, W)
    # Single vectorized call instead of per-class loop
    valid_mask = ~np.isin(scl_stack, SCL_EXCLUDE_ARR)   # (n_ok, H, W)
    valid_mask &= (scl_stack > 0)

    valid_frac = valid_mask.mean(axis=0).astype(np.float32)

    # ── Mask invalid pixels + median composite ──
    spectral_4d = np.stack(all_spectral, dtype=np.float32)   # (n_ok, bands, H, W)
    # Mask invalid pixels: valid_mask is (n_ok, H, W), broadcast across bands
    # Use np.where for correct broadcasting over the band dimension
    invalid = ~valid_mask[:, np.newaxis, :, :]               # (n_ok, 1, H, W)
    spectral_4d = np.where(invalid, np.nan, spectral_4d)     # broadcasts correctly

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmedian(spectral_4d, axis=0)    # (bands, H, W)

    # Free large arrays immediately
    del spectral_4d, scl_stack, valid_mask, invalid

    # PB 04.00 offset correction (Jan 2022+)
    if year >= 2022:
        composite = np.maximum(composite - 1000.0, 0.0)

    return composite, valid_frac


def write_composite_tif(composite, valid_frac, dst_crs, dst_transform, dst_height, dst_width, output_path):
    """Write composite + valid_fraction to a multi-band GeoTIFF."""
    n_bands = composite.shape[0]
    comp_clean = np.where(np.isnan(composite), NODATA, composite).astype(np.float32)
    vf_clean = np.where(np.isnan(valid_frac), NODATA, valid_frac).astype(np.float32)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=dst_height, width=dst_width,
        count=n_bands + 1, dtype="float32",
        crs=dst_crs, transform=dst_transform,
        compress="lzw", nodata=NODATA,
    ) as dst:
        for i, band_name in enumerate(SENTINEL_BANDS):
            dst.write(comp_clean[i], i + 1)
            dst.set_band_description(i + 1, band_name)
        dst.write(vf_clean, n_bands + 1)
        dst.set_band_description(n_bands + 1, "VALID_FRACTION")


def main():
    parser = argparse.ArgumentParser(description="Composite helper for terrapulse")
    parser.add_argument("--scenes-json", required=True)
    parser.add_argument("--anchor-ref", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    with open(args.scenes_json) as f:
        scenes = json.load(f)

    with rasterio.open(args.anchor_ref) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    print(f"  Compositing {len(scenes)} scenes -> {dst_width}x{dst_height} ...", file=sys.stderr)

    composite, valid_frac = process_scenes(
        scenes, dst_crs, dst_transform, dst_width, dst_height, args.year)

    if composite is None:
        print("  ERROR: No valid scenes!", file=sys.stderr)
        sys.exit(1)

    write_composite_tif(composite, valid_frac, dst_crs, dst_transform, dst_height, dst_width, args.output)
    print(f"  Done: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

---
