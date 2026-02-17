#!/usr/bin/env python3
"""Export V4 production MLP to ONNX for the Rust deploy pipeline.

Replaces the old 5-fold models with the V4 single-seed production model.
Output goes to data/pipeline_output/models/onnx/
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

V4_MODELS = os.path.join(ROOT, "data", "cities", "models_v4")
ONNX_DIR = os.path.join(ROOT, "data", "pipeline_output", "models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

N_CLASSES = 6


# -- MLP architecture (must match run_multi_city_pipeline_v4.py) --
class PlainBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.30):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.norm(F.silu(self.linear(x))))


class PlainMLP(nn.Module):
    def __init__(self, in_features, n_classes=N_CLASSES, hidden=2048,
                 n_layers=5, dropout=0.30):
        super().__init__()
        layers = [PlainBlock(in_features, hidden, dropout)]
        for _ in range(n_layers - 1):
            layers.append(PlainBlock(hidden, hidden, dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return torch.softmax(self.head(self.backbone(x)), dim=-1)


def main():
    print("=" * 60)
    print("V4 ONNX Export")
    print("=" * 60)

    # Load meta
    with open(os.path.join(V4_MODELS, "mlp_meta.json")) as f:
        meta = json.load(f)
    n_features = meta["n_features"]
    mlp_cols = meta["feature_cols"]
    print(f"  V4 MLP: {n_features} features, depth={meta['arch']['depth']}, "
          f"width={meta['arch']['width']}")

    # ---- Export MLP ----
    print(f"\nExporting MLP...")
    pt_path = os.path.join(V4_MODELS, "mlp_seed0.pt")
    net = PlainMLP(n_features)
    net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    net.eval()

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

    # Remove old fold files (1-4)
    for fold in range(1, 5):
        old = os.path.join(ONNX_DIR, f"mlp_fold_{fold}.onnx")
        old_data = old + ".data"
        if os.path.exists(old):
            os.remove(old)
            print(f"  Removed old: mlp_fold_{fold}.onnx")
        if os.path.exists(old_data):
            os.remove(old_data)
            print(f"  Removed old: mlp_fold_{fold}.onnx.data")

    # ---- Export scaler ----
    print(f"\nExporting scaler...")
    pkl_path = os.path.join(V4_MODELS, "mlp_scaler.pkl")
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

    print(f"\n{'='*60}")
    print("DONE. Files written to:")
    for name in sorted(os.listdir(ONNX_DIR)):
        if name.startswith("mlp"):
            size = os.path.getsize(os.path.join(ONNX_DIR, name))
            print(f"  {name:40s} {size/1024:>8.0f} KB")
    print(f"{'='*60}")
    print("\nNOTE: Update terrapulse/src/config.rs N_FOLDS from 5 -> 1")
    print("      Then: cd terrapulse && cargo build --release")


if __name__ == "__main__":
    main()
