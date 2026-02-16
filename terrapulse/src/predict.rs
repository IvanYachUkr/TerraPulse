use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;

use crate::config::{N_CLASSES, N_FOLDS};

/// Scaler parameters (mean + scale) for StandardScaler transform.
#[derive(serde::Deserialize)]
pub struct ScalerParams {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub n_features: usize,
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
                if s.abs() < 1e-12 { 0.0 } else { (v - m) / s }
            })
            .collect()
    }
}

/// A loaded ensemble of ONNX models for one model type (tree or MLP).
pub struct OnnxEnsemble {
    /// For tree: sessions[fold][class_idx], for MLP: sessions[fold] (single)
    sessions: Vec<Vec<Session>>,
    pub n_folds: usize,
    pub model_type: ModelType,
}

#[derive(Clone, Copy)]
pub enum ModelType {
    Tree,
    Mlp,
}

impl OnnxEnsemble {
    /// Load tree ensemble: 5 folds x 6 class models.
    pub fn load_trees(onnx_dir: &Path) -> Result<Self> {
        let mut sessions = Vec::with_capacity(N_FOLDS);
        for fold in 0..N_FOLDS {
            let mut fold_sessions = Vec::with_capacity(N_CLASSES);
            for ci in 0..N_CLASSES {
                let path = onnx_dir.join(format!("tree_fold_{fold}_class{ci}.onnx"));
                let session = Session::builder()?
                    .with_intra_threads(1)?
                    .commit_from_file(&path)
                    .with_context(|| format!("Cannot load ONNX: {}", path.display()))?;
                fold_sessions.push(session);
            }
            sessions.push(fold_sessions);
        }
        Ok(Self {
            sessions,
            n_folds: N_FOLDS,
            model_type: ModelType::Tree,
        })
    }

    /// Load MLP ensemble: 5 folds, 1 model each.
    pub fn load_mlps(onnx_dir: &Path) -> Result<Self> {
        let mut sessions = Vec::with_capacity(N_FOLDS);
        for fold in 0..N_FOLDS {
            let path = onnx_dir.join(format!("mlp_fold_{fold}.onnx"));
            let session = Session::builder()?
                .with_intra_threads(1)?
                .commit_from_file(&path)
                .with_context(|| format!("Cannot load ONNX: {}", path.display()))?;
            sessions.push(vec![session]);
        }
        Ok(Self {
            sessions,
            n_folds: N_FOLDS,
            model_type: ModelType::Mlp,
        })
    }

    /// Run inference for all cells. Returns [n_cells, N_CLASSES] averaged across folds.
    ///
    /// `features`: [n_cells][n_features_for_this_model] row-major.
    pub fn predict(&mut self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_cells = features.len();
        if n_cells == 0 {
            return Ok(Vec::new());
        }

        // Accumulate predictions across folds
        let mut accum = vec![vec![0.0f64; N_CLASSES]; n_cells];

        for fold in 0..self.n_folds {
            match self.model_type {
                ModelType::Tree => {
                    // Tree: run each class model separately
                    for ci in 0..N_CLASSES {
                        let session = &mut self.sessions[fold][ci];
                        let predictions = run_onnx_batch(session, features, "X")?;
                        for (row_idx, pred) in predictions.iter().enumerate() {
                            accum[row_idx][ci] += pred[0] as f64;
                        }
                    }
                }
                ModelType::Mlp => {
                    let session = &mut self.sessions[fold][0];
                    let predictions = run_onnx_batch(session, features, "X")?;
                    for (row_idx, pred) in predictions.iter().enumerate() {
                        for ci in 0..N_CLASSES {
                            accum[row_idx][ci] += pred[ci] as f64;
                        }
                    }
                }
            }
        }

        // Average across folds and clip
        let inv_folds = 1.0 / self.n_folds as f64;
        let result: Vec<Vec<f32>> = accum
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|v| {
                        let avg = (v * inv_folds) as f32;
                        match self.model_type {
                            ModelType::Tree => avg.clamp(0.0, 100.0),
                            ModelType::Mlp => avg, // probabilities, already 0-1
                        }
                    })
                    .collect()
            })
            .collect();

        Ok(result)
    }

    /// Run a single fold's model(s). Returns [n_cells, N_CLASSES] for that fold.
    pub fn predict_single_fold(&mut self, fold: usize, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_cells = features.len();
        if n_cells == 0 {
            return Ok(Vec::new());
        }

        let mut result = vec![vec![0.0f32; N_CLASSES]; n_cells];

        match self.model_type {
            ModelType::Tree => {
                for ci in 0..N_CLASSES {
                    let session = &mut self.sessions[fold][ci];
                    let predictions = run_onnx_batch(session, features, "X")?;
                    for (ri, pred) in predictions.iter().enumerate() {
                        result[ri][ci] = pred[0].clamp(0.0, 100.0);
                    }
                }
            }
            ModelType::Mlp => {
                let session = &mut self.sessions[fold][0];
                let predictions = run_onnx_batch(session, features, "X")?;
                for (ri, pred) in predictions.iter().enumerate() {
                    for ci in 0..N_CLASSES {
                        result[ri][ci] = pred[ci];
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Run ONNX session on a batch of inputs, returning [n_rows][n_outputs].
fn run_onnx_batch(
    session: &mut Session,
    features: &[Vec<f32>],
    input_name: &str,
) -> Result<Vec<Vec<f32>>> {
    let n_rows = features.len();
    let n_cols = features[0].len();

    // Flatten to contiguous array
    let mut flat: Vec<f32> = Vec::with_capacity(n_rows * n_cols);
    for row in features {
        flat.extend_from_slice(row);
    }

    // Create ONNX tensor — ort 2.x needs (shape, Vec<T>)
    let input_tensor =
        ort::value::Tensor::from_array(([n_rows, n_cols], flat))?;

    let outputs = session.run(ort::inputs![input_name => input_tensor])?;

    // Extract output — try_extract_tensor returns (&Shape, &[f32])
    // Shape derefs to [i64]
    let output = &outputs[0];
    let (tensor_shape, tensor_data) = output.try_extract_tensor::<f32>()?;

    let out_cols = if tensor_shape.len() > 1 { tensor_shape[1] as usize } else { 1 };

    let mut result = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let start = i * out_cols;
        let end = start + out_cols;
        result.push(tensor_data[start..end].to_vec());
    }

    Ok(result)
}

