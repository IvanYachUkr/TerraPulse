#!/usr/bin/env python3
"""
Export best V10 BOHB model (#7, trial 77, T_1024_512_256_64, gelu) to ONNX
for the Rust terrapulse inference pipeline.

Key difference from V8: V10 model uses PlainBlock with configurable activation
(gelu) and log_softmax output. For ONNX export, we override forward() to use
softmax so the Rust pipeline gets probabilities directly.

Usage:
    .venv/Scripts/python scripts/export_best_v10.py
"""

import os, sys, json, pickle, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_mlp_overnight_v4 import PlainBlock

# =====================================================================
# Model class — modified forward to output softmax (not log_softmax)
# so ONNX output matches what Rust predict.rs expects.
# =====================================================================
class TaperedMLP_Export(nn.Module):
    """Same architecture as V10 TaperedMLP but with softmax output for ONNX."""
    def __init__(self, in_features, n_classes, widths, dropout=0.15,
                 activation="gelu", input_dropout=0.05, norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = []
        prev_dim = in_features
        for w in widths:
            layers.append(PlainBlock(prev_dim, w, dropout, activation, norm_type))
            prev_dim = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        # Use softmax (not log_softmax) for ONNX export compatibility
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)


def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("ONNX EXPORT: Best V10 BOHB Model (#7)")
    print(sep)

    V10_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v10_bohb")
    ONNX_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline_output", "models", "onnx")

    # --- Identify the model ---
    trials = [json.loads(l) for l in open(os.path.join(V10_DIR, "trial_log.jsonl")) if l.strip()][20:]
    ranked = sorted(trials, key=lambda t: t["combined"], reverse=True)
    seen, topN = set(), []
    for t in ranked:
        k = json.dumps(t["config"], sort_keys=True)
        if k not in seen:
            seen.add(k)
            topN.append(t)
        if len(topN) >= 7:
            break

    best = topN[6]  # #7 (0-indexed = 6)
    print(f"  Trial:       {best['trial']}")
    print(f"  Architecture:{best['arch']}")
    print(f"  Activation:  {best['config']['activation']}")
    print(f"  Widths:      {best['widths']}")
    print(f"  Dropout:     {best['config']['dropout']:.4f}")
    print(f"  Input DO:    {best['config']['input_dropout']:.4f}")
    print(f"  Val Combined:{best['combined']:.4f}")

    # --- Load columns and scaler ---
    cols_path = os.path.join(V10_DIR, "mlp_cols.json")
    with open(cols_path) as f:
        mlp_cols = json.load(f)
    n_features = len(mlp_cols)
    n_classes = 7
    print(f"  Features:    {n_features}")

    scaler_path = os.path.join(V10_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --- Build model and load weights ---
    net = TaperedMLP_Export(
        n_features, n_classes, best["widths"],
        dropout=best["config"]["dropout"],
        activation=best["config"]["activation"],
        input_dropout=best["config"]["input_dropout"],
    )
    pt_path = os.path.join(V10_DIR, f"trial_{best['trial']}_{best['arch']}.pt")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters:  {n_params:,}")
    print(f"  Loaded from: {pt_path}")

    # --- Back up existing V8 ONNX files ---
    print(f"\n  Backing up existing ONNX files...")
    for fname in ["mlp_fold_0.onnx", "mlp_scaler_0.json", "mlp_cols.json"]:
        src = os.path.join(ONNX_DIR, fname)
        dst = os.path.join(ONNX_DIR, f"{fname}.bak_v8")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"    {fname} -> {fname}.bak_v8")
        elif os.path.exists(dst):
            print(f"    {fname}.bak_v8 already exists — skip")

    # --- Export ONNX ---
    print(f"\n  Exporting to ONNX...")
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

    # --- Export scaler as JSON ---
    print(f"\n  Exporting scaler...")
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    json_path = os.path.join(ONNX_DIR, "mlp_scaler_0.json")
    with open(json_path, "w") as f:
        json.dump(scaler_data, f)
    print(f"  Scaler: {scaler.n_features_in_} features -> {json_path}")

    # --- Export column list ---
    cols_out = os.path.join(ONNX_DIR, "mlp_cols.json")
    with open(cols_out, "w") as f:
        json.dump(mlp_cols, f)
    print(f"  Columns: {len(mlp_cols)} -> mlp_cols.json")

    # --- Validate ONNX ---
    print(f"\n  Validating ONNX output...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        rng = np.random.RandomState(42)
        X_test = rng.randn(100, n_features).astype(np.float32)

        with torch.no_grad():
            py_pred = net(torch.tensor(X_test)).numpy()
        onnx_pred = sess.run(None, {"X": X_test})[0]

        max_diff = np.max(np.abs(py_pred - onnx_pred))
        row_sums = onnx_pred.sum(axis=1)
        all_positive = (onnx_pred >= 0).all()

        print(f"  Max abs diff:  {max_diff:.8f} [{'OK' if max_diff < 1e-4 else 'MISMATCH'}]")
        print(f"  Row sums:      [{row_sums.min():.6f}, {row_sums.max():.6f}] (should be ~1.0)")
        print(f"  All positive:  {all_positive}")

        if max_diff >= 1e-4:
            print(f"  WARNING: ONNX validation failed! max_diff={max_diff}")
            sys.exit(1)
    except ImportError:
        print("  WARNING: onnxruntime not available, skipping validation")

    # --- Summary ---
    print(f"\n{sep}")
    print("ONNX export complete. Files:")
    for name in sorted(os.listdir(ONNX_DIR)):
        size = os.path.getsize(os.path.join(ONNX_DIR, name))
        print(f"  {name:45s} {size/1024:>8.0f} KB")
    print(sep)


if __name__ == "__main__":
    main()
