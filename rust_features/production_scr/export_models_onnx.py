#!/usr/bin/env python3
"""Export trained pipeline models to ONNX format.

Converts:
  - LightGBM MultiOutputRegressor (5 folds) -> tree_fold_N_classC.onnx
  - PyTorch MLP (5 folds) -> mlp_fold_N.onnx
  - StandardScaler (5 folds) -> mlp_scaler_N.json
  - Feature column lists -> tree_cols.json, mlp_cols.json

All outputs go to  data/pipeline_output/models/onnx/

Usage:
    python export_models_onnx.py
"""

import json
import os
import pickle
import sys
import numpy as np

# -------------------------------------------------------------------
# paths
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _find_root():
    d = SCRIPT_DIR
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "data")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot find project root")

ROOT = _find_root()
MODELS_DIR = os.path.join(ROOT, "data", "pipeline_output", "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

N_FOLDS = 5
N_CLASSES = 6

# -------------------------------------------------------------------
# 1) LightGBM -> ONNX (per sub-estimator)
# -------------------------------------------------------------------
def export_trees():
    from onnxmltools.convert import convert_lightgbm
    from onnxconverter_common import FloatTensorType
    import onnx
    import onnxruntime as ort

    with open(os.path.join(MODELS_DIR, "tree_meta.json")) as f:
        meta = json.load(f)
    n_features = meta["n_features"]
    tree_cols = meta["feature_cols"]

    # Save column list for Rust
    with open(os.path.join(ONNX_DIR, "tree_cols.json"), "w") as f:
        json.dump(tree_cols, f)

    print(f"Tree model: {n_features} features -> {N_CLASSES} outputs")

    for fold in range(N_FOLDS):
        pkl_path = os.path.join(MODELS_DIR, f"tree_fold_{fold}.pkl")

        with open(pkl_path, "rb") as f:
            multi_model = pickle.load(f)

        # MultiOutputRegressor wraps N_CLASSES LGBMRegressor instances
        sub_estimators = multi_model.estimators_
        assert len(sub_estimators) == N_CLASSES, (
            f"Expected {N_CLASSES} sub-estimators, got {len(sub_estimators)}")

        rng = np.random.RandomState(42)
        X_test = rng.randn(100, n_features).astype(np.float32)
        py_pred = np.clip(multi_model.predict(X_test), 0, 100).astype(np.float32)

        onnx_preds = np.zeros_like(py_pred)

        for ci in range(N_CLASSES):
            onnx_path = os.path.join(
                ONNX_DIR, f"tree_fold_{fold}_class{ci}.onnx")

            onnx_model = convert_lightgbm(
                sub_estimators[ci],
                initial_types=[("X", FloatTensorType([None, n_features]))],
                target_opset=15,
            )
            onnx.save(onnx_model, onnx_path)

            sess = ort.InferenceSession(onnx_path)
            onnx_out = sess.run(None, {"X": X_test})[0].flatten()
            onnx_preds[:, ci] = onnx_out

        onnx_preds = np.clip(onnx_preds, 0, 100).astype(np.float32)
        max_diff = np.max(np.abs(py_pred - onnx_preds))
        ok = np.allclose(py_pred, onnx_preds, rtol=1e-3, atol=1e-3)
        status = "OK" if ok else "MISMATCH"
        print(f"  Fold {fold}: max_diff={max_diff:.6f} [{status}]")
        if not ok:
            print(f"  WARNING: fold {fold} exceeds tolerance!")

    print()


# -------------------------------------------------------------------
# 2) MLP -> ONNX
# -------------------------------------------------------------------
def export_mlps():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import onnxruntime as ort

    with open(os.path.join(MODELS_DIR, "mlp_meta.json")) as f:
        meta = json.load(f)
    n_features = meta["n_features"]
    mlp_cols = meta["feature_cols"]

    # Save column list for Rust
    with open(os.path.join(ONNX_DIR, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    print(f"MLP model: {n_features} features -> {N_CLASSES} outputs")

    # Rebuild model class (must match pipeline.py exactly)
    class PlainBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.15):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.norm = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            return self.dropout(self.norm(F.silu(self.linear(x))))

    class PlainMLP(nn.Module):
        def __init__(self, in_features, n_classes=N_CLASSES, hidden=1024,
                     n_layers=5, dropout=0.15):
            super().__init__()
            layers = [PlainBlock(in_features, hidden, dropout)]
            for _ in range(n_layers - 1):
                layers.append(PlainBlock(hidden, hidden, dropout))
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, n_classes)
        def forward(self, x):
            # For ONNX export: produce softmax probabilities directly
            return torch.softmax(self.head(self.backbone(x)), dim=-1)

    for fold in range(N_FOLDS):
        pt_path = os.path.join(MODELS_DIR, f"mlp_fold_{fold}.pt")
        onnx_path = os.path.join(ONNX_DIR, f"mlp_fold_{fold}.onnx")

        net = PlainMLP(n_features)
        net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        net.eval()

        # Export
        dummy = torch.randn(1, n_features)
        torch.onnx.export(
            net, dummy, onnx_path,
            input_names=["X"],
            output_names=["probabilities"],
            dynamic_axes={"X": {0: "batch"}, "probabilities": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )

        # Validate
        sess = ort.InferenceSession(onnx_path)
        rng = np.random.RandomState(42)
        X_test = rng.randn(100, n_features).astype(np.float32)

        with torch.no_grad():
            py_pred = net(torch.tensor(X_test)).numpy()
        onnx_pred = sess.run(None, {"X": X_test})[0]

        max_diff = np.max(np.abs(py_pred - onnx_pred))
        ok = np.allclose(py_pred, onnx_pred, rtol=1e-4, atol=1e-5)
        status = "OK" if ok else "MISMATCH"
        print(f"  Fold {fold}: max_diff={max_diff:.8f} [{status}]")

    print()


# -------------------------------------------------------------------
# 3) StandardScaler -> JSON
# -------------------------------------------------------------------
def export_scalers():
    print("Scalers:")
    for fold in range(N_FOLDS):
        pkl_path = os.path.join(MODELS_DIR, f"mlp_scaler_{fold}.pkl")
        json_path = os.path.join(ONNX_DIR, f"mlp_scaler_{fold}.json")

        with open(pkl_path, "rb") as f:
            scaler = pickle.load(f)

        data = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_features": int(scaler.n_features_in_),
        }
        with open(json_path, "w") as f:
            json.dump(data, f)
        print(f"  Fold {fold}: {scaler.n_features_in_} features -> {json_path}")

    print()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Model Export")
    print("=" * 60)
    print(f"  Source: {MODELS_DIR}")
    print(f"  Output: {ONNX_DIR}")
    print()

    export_trees()
    export_mlps()
    export_scalers()

    print("=" * 60)
    print("All models exported to ONNX.")
    print(f"  Output: {ONNX_DIR}")
    print("=" * 60)
