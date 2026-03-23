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

        // Read nan_fraction column (data quality tracking)
        let nan_frac_idx = col_index.get("nan_fraction");
        let nan_fractions: Vec<f32> = if let Some(&idx) = nan_frac_idx {
            all_rows.iter().map(|row| row[idx]).collect()
        } else {
            vec![0.0; n_cells] // no quality data → assume all good
        };

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
            let mut nodata_count = 0usize;
            let mut low_data_count = 0usize;
            for (i, row) in mlp_preds.iter().enumerate() {
                let nan_frac = nan_fractions[i];
                let mut cell_map = serde_json::Map::with_capacity(CLASS_NAMES.len() + 1);

                if nan_frac > 0.5 {
                    // >50% features were NaN — mark as nodata, skip prediction
                    cell_map.insert("_quality".to_string(), serde_json::json!("nodata"));
                    nodata_count += 1;
                } else {
                    // Include predictions
                    for (j, &val) in row.iter().enumerate() {
                        let rounded = (val * 10000.0).round() / 10000.0;
                        cell_map.insert(CLASS_NAMES[j].to_string(), serde_json::json!(rounded));
                    }
                    // Add quality flag
                    if nan_frac > 0.0 {
                        cell_map.insert("_quality".to_string(), serde_json::json!("low_data"));
                        low_data_count += 1;
                    } else {
                        cell_map.insert("_quality".to_string(), serde_json::json!("good"));
                    }
                }
                json_map.insert(i.to_string(), serde_json::Value::Object(cell_map));
            }
            let json_str = serde_json::to_string(&serde_json::Value::Object(json_map))?;
            std::fs::write(&json_out, json_str)?;
            println!(
                "  Wrote json {} ({} cells: {} good, {} low_data, {} nodata)",
                json_out.display(), n_cells,
                n_cells - low_data_count - nodata_count,
                low_data_count, nodata_count,
            );
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
        if y == 2020 || y == 2021 {
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
