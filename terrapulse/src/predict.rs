use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;

use crate::config::N_CLASSES;

/// Model configuration loaded from model_config.json.
#[derive(serde::Deserialize)]
pub struct ModelConfig {
    pub label_threshold: f32,
    #[serde(default = "default_n_classes")]
    #[allow(dead_code)]
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
