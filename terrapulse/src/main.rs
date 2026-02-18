mod config;
mod download;
mod extract;
mod features;
mod parquet_io;
mod predict;
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
        #[arg(long, num_args = 4)]
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
        #[arg(long, num_args = 4)]
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
            run_extract(&year_pairs, &region, &raw_dir, &features_dir, min_valid_frac)?;
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
                &bbox, epsg, &years, &region, &data_dir, &anchor_ref,
                &models_dir, min_valid_frac,
                skip_download, skip_extract, skip_predict,
            ).await?;
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
            prev_year, curr_year, region, raw_dir, features_dir, min_valid_frac,
        )?;
    }

    println!("\nTotal extraction time: {:.1}s", t0.elapsed().as_secs_f64());
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

    // ---- Load column lists ----
    let tree_cols: Vec<String> = {
        let data = std::fs::read_to_string(models_dir.join("tree_cols.json"))?;
        serde_json::from_str(&data)?
    };
    let mlp_cols: Vec<String> = {
        let data = std::fs::read_to_string(models_dir.join("mlp_cols.json"))?;
        serde_json::from_str(&data)?
    };
    println!("Tree features: {}, MLP features: {}", tree_cols.len(), mlp_cols.len());

    // ---- Load models ----
    println!("Loading tree models...");
    let t_load = Instant::now();
    let mut tree_ensemble = predict::OnnxEnsemble::load_trees(models_dir)?;
    println!("  Loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    println!("Loading MLP models...");
    let t_load = Instant::now();
    let mut mlp_ensemble = predict::OnnxEnsemble::load_mlps(models_dir)?;
    println!("  Loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    // ---- Load scalers ----
    let scalers: Vec<predict::ScalerParams> = (0..config::N_FOLDS)
        .map(|i| {
            let path = models_dir.join(format!("mlp_scaler_{i}.json"));
            predict::ScalerParams::load(&path)
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Loaded {} scalers", scalers.len());

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

        // ---- Extract tree features ----
        let tree_indices: Vec<usize> = tree_cols
            .iter()
            .map(|c| {
                *col_index
                    .get(c.as_str())
                    .with_context(|| format!("Tree col '{}' not found in parquet", c))
                    .unwrap()
            })
            .collect();

        let tree_features: Vec<Vec<f32>> = all_rows
            .iter()
            .map(|row| {
                tree_indices
                    .iter()
                    .map(|&i| {
                        let v = row[i];
                        if v.is_finite() { v } else { 0.0 }
                    })
                    .collect()
            })
            .collect();

        // ---- Extract MLP features + scale ----
        let mlp_indices: Vec<usize> = mlp_cols
            .iter()
            .map(|c| {
                *col_index
                    .get(c.as_str())
                    .with_context(|| format!("MLP col '{}' not found in parquet", c))
                    .unwrap()
            })
            .collect();

        let mlp_features_raw: Vec<Vec<f32>> = all_rows
            .iter()
            .map(|row| {
                mlp_indices
                    .iter()
                    .map(|&i| {
                        let v = row[i];
                        if v.is_finite() { v } else { 0.0 }
                    })
                    .collect()
            })
            .collect();


        // ---- Run tree prediction ----
        println!("  Running tree inference...");
        let t_pred = Instant::now();
        let tree_preds = tree_ensemble.predict(&tree_features)?;
        println!(
            "  Tree done: {} cells in {:.2}s",
            n_cells,
            t_pred.elapsed().as_secs_f64()
        );

        // ---- Run MLP prediction (per-fold scaler, matching Python) ----
        println!("  Running MLP inference...");
        let t_pred = Instant::now();
        let n_classes = config::N_CLASSES;
        let mut mlp_accum = vec![vec![0.0f64; n_classes]; n_cells];
        for fold in 0..config::N_FOLDS {
            // Apply this fold's scaler
            let scaled: Vec<Vec<f32>> = mlp_features_raw
                .iter()
                .map(|row| scalers[fold].transform(row))
                .collect();
            // Run this fold's model
            let fold_preds = mlp_ensemble.predict_single_fold(fold, &scaled)?;
            for (ri, pred) in fold_preds.iter().enumerate() {
                for ci in 0..n_classes {
                    mlp_accum[ri][ci] += pred[ci] as f64;
                }
            }
        }
        let inv_folds = 1.0 / config::N_FOLDS as f64;
        let mlp_preds: Vec<Vec<f32>> = mlp_accum
            .into_iter()
            .map(|row| row.into_iter().map(|v| (v * inv_folds) as f32).collect())
            .collect();
        println!(
            "  MLP done: {} cells in {:.2}s",
            n_cells,
            t_pred.elapsed().as_secs_f64()
        );

        // ---- Save predictions ----
        let tree_out = output_dir.join(format!("pred_tree_{yp}.parquet"));
        parquet_io::write_predictions_parquet(&tree_out, &CLASS_NAMES, &tree_preds, "tree")?;
        println!("  Wrote {}", tree_out.display());

        let mlp_out = output_dir.join(format!("pred_mlp_{yp}.parquet"));
        parquet_io::write_predictions_parquet(&mlp_out, &CLASS_NAMES, &mlp_preds, "mlp")?;
        println!("  Wrote {}", mlp_out.display());
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

    // Resolve the composite helper script (relative to the binary or provided)
    let helper_script = find_helper_script()?;

    let bbox_arr: [f64; 4] = [bbox[0], bbox[1], bbox[2], bbox[3]];

    // Build HTTP client with generous timeouts for large COG downloads
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    println!("TerraPulse Download");
    println!("  Region: {region}");
    println!("  BBOX: [{}, {}, {}, {}]", bbox[0], bbox[1], bbox[2], bbox[3]);
    println!("  EPSG: {epsg}");
    println!("  Years: {:?}", years);
    println!("  Helper: {}", helper_script.display());
    println!();

    for &year in years {
        println!("--- Year: {year} ---");
        download::download_year(
            &client,
            bbox_arr,
            epsg,
            year,
            region,
            raw_dir,
            anchor_ref,
            &helper_script,
        )
        .await?;
    }

    println!(
        "\nTotal download time: {:.1}s",
        t0.elapsed().as_secs_f64()
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

    println!("\n{sep}");
    println!("Pipeline complete in {:.1}s", t0.elapsed().as_secs_f64());
    println!("{sep}");

    Ok(())
}

/// Find the composite.py helper script.
/// Looks for it relative to the executable, then in the terrapulse/helpers dir.
fn find_helper_script() -> Result<PathBuf> {
    // Try next to the executable
    if let Ok(exe) = std::env::current_exe() {
        let dir = exe.parent().unwrap_or(Path::new("."));
        let candidate = dir.join("helpers").join("composite.py");
        if candidate.exists() {
            return Ok(candidate);
        }
        // Try sibling of target dir (source layout)
        let candidate = dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join("helpers").join("composite.py"));
        if let Some(p) = candidate {
            if p.exists() {
                return Ok(p);
            }
        }
    }

    // Try relative to CWD
    let cwd_candidate = PathBuf::from("terrapulse/helpers/composite.py");
    if cwd_candidate.exists() {
        return Ok(cwd_candidate);
    }

    anyhow::bail!(
        "Cannot find helpers/composite.py. \
         Place it next to the terrapulse executable or run from the project root."
    );
}
