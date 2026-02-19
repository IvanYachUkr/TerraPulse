#!/usr/bin/env python3
"""Export V6 sweep winner (T_1024_512_256_64_mixup) to ONNX for the Rust deploy pipeline.

Architecture: Tapered MLP [1764 → 1024 → 512 → 256 → 64 → 7]
Training mode: mixup_only
Output: softmax probabilities for 7 classes
"""

import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

# Paths
V6_SWEEP_DIR = os.path.join(ROOT, "data", "cities", "models_v6_sweep", "20260219_035147")
ONNX_DIR = os.path.join(ROOT, "data", "pipeline_output", "models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

N_CLASSES = 7
MODEL_NAME = "T_1024_512_256_64_mixup"
WIDTHS = [1024, 512, 256, 64]
DROPOUT = 0.15
INPUT_DROPOUT = 0.05


# -- Architecture (must match sweep_mlp_v6.py) --
class PlainBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.linear(x))))


class TaperedMLP(nn.Module):
    def __init__(self, in_features, n_classes, widths, dropout=0.15,
                 input_dropout=0.05):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = []
        prev_dim = in_features
        for w in widths:
            layers.append(PlainBlock(prev_dim, w, dropout))
            prev_dim = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        # For ONNX export: softmax output (probabilities)
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)


def main():
    sep = "=" * 60
    print(sep)
    print("V6 ONNX Export")
    print(sep)

    # Load column list
    cols_path = os.path.join(V6_SWEEP_DIR, "mlp_cols.json")
    with open(cols_path) as f:
        mlp_cols = json.load(f)
    n_features = len(mlp_cols)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {N_CLASSES}")
    print(f"  Widths: {WIDTHS}")

    # Build model
    net = TaperedMLP(n_features, N_CLASSES, WIDTHS, DROPOUT, INPUT_DROPOUT)

    # Load weights
    pt_path = os.path.join(V6_SWEEP_DIR, f"{MODEL_NAME}.pt")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    print(f"  Loaded weights from {pt_path}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Export ONNX ----
    print(f"\nExporting MLP to ONNX...")
    onnx_path = os.path.join(ONNX_DIR, "mlp_fold_0.onnx")
    dummy = torch.randn(1, n_features)
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["X"],
        output_names=["probabilities"],
        dynamic_axes={"X": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Wrote: {onnx_path} ({onnx_size:.1f} MB)")

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    rng = np.random.RandomState(42)
    X_test = rng.randn(100, n_features).astype(np.float32)
    with torch.no_grad():
        py_pred = net(torch.tensor(X_test)).numpy()
    onnx_pred = sess.run(None, {"X": X_test})[0]
    max_diff = np.max(np.abs(py_pred - onnx_pred))
    print(f"  ONNX max_diff: {max_diff:.8f} "
          f"[{'OK' if max_diff < 1e-4 else 'MISMATCH'}]")

    # Check output sums to ~1
    row_sums = onnx_pred.sum(axis=1)
    print(f"  Row sums: min={row_sums.min():.6f} max={row_sums.max():.6f} "
          f"(should be ~1.0)")

    # Remove old fold files (1-4)
    for fold in range(1, 5):
        for ext in [".onnx", ".onnx.data"]:
            old = os.path.join(ONNX_DIR, f"mlp_fold_{fold}{ext}")
            if os.path.exists(old):
                os.remove(old)
                print(f"  Removed old: mlp_fold_{fold}{ext}")

    # ---- Export scaler ----
    print(f"\nExporting scaler...")
    pkl_path = os.path.join(V6_SWEEP_DIR, "scaler.pkl")
    with open(pkl_path, "rb") as f:
        scaler = pickle.load(f)

    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    json_path = os.path.join(ONNX_DIR, "mlp_scaler_0.json")
    with open(json_path, "w") as f:
        json.dump(scaler_data, f)
    print(f"  Scaler: {scaler.n_features_in_} features")

    # Remove old scaler files (1-4)
    for fold in range(1, 5):
        old = os.path.join(ONNX_DIR, f"mlp_scaler_{fold}.json")
        if os.path.exists(old):
            os.remove(old)
            print(f"  Removed old: mlp_scaler_{fold}.json")

    # ---- Export column list ----
    print(f"\nExporting mlp_cols.json ({len(mlp_cols)} features)...")
    with open(os.path.join(ONNX_DIR, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    # Remove old tree models
    print(f"\nCleaning up old tree models...")
    removed = 0
    for fname in os.listdir(ONNX_DIR):
        if fname.startswith("tree_"):
            os.remove(os.path.join(ONNX_DIR, fname))
            removed += 1
    if removed:
        print(f"  Removed {removed} old tree model files")

    # Remove tree_cols.json
    tree_cols = os.path.join(ONNX_DIR, "tree_cols.json")
    if os.path.exists(tree_cols):
        os.remove(tree_cols)
        print(f"  Removed tree_cols.json")

    print(f"\n{sep}")
    print("DONE. Files in ONNX dir:")
    for name in sorted(os.listdir(ONNX_DIR)):
        size = os.path.getsize(os.path.join(ONNX_DIR, name))
        print(f"  {name:40s} {size/1024:>8.0f} KB")
    print(sep)


if __name__ == "__main__":
    main()
