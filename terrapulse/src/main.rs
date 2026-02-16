mod config;
mod download;
mod parquet_io;
mod predict;
mod stac;

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
    }

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
