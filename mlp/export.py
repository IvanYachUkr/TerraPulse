"""
ONNX export and scaler serialisation for the Rust deploy pipeline.
"""

import json
import os
from typing import List

import numpy as np
import torch

from .config import ONNX_DIR


def export_onnx(net: torch.nn.Module, n_features: int, out_path: str):
    """Export model to ONNX with softmax output for Rust inference.

    Uses the export_forward method (softmax instead of log-softmax).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.eval()

    # Swap forward to export version
    original_forward = net.forward
    net.forward = net.export_forward

    dummy = torch.randn(1, n_features)
    torch.onnx.export(
        net, dummy, out_path,
        input_names=["X"],
        output_names=["probabilities"],
        dynamic_axes={"X": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Restore
    net.forward = original_forward

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ONNX saved: {out_path} ({size_mb:.1f} MB)")


def export_scaler(scaler, out_path: str):
    """Save StandardScaler parameters as JSON for Rust inference."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    with open(out_path, "w") as f:
        json.dump(scaler_data, f)
    print(f"  Scaler saved: {out_path} ({scaler.n_features_in_} features)")


def export_columns(columns: List[str], out_path: str):
    """Save the ordered feature column list as JSON."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(columns, f)
    print(f"  Columns saved: {out_path} ({len(columns)} features)")


def validate_onnx(onnx_path: str, net: torch.nn.Module, n_features: int):
    """Cross-check PyTorch vs ONNX outputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    rng = np.random.RandomState(42)
    X_test = rng.randn(100, n_features).astype(np.float32)

    # PyTorch (softmax)
    net.eval()
    with torch.no_grad():
        py_pred = net.export_forward(torch.tensor(X_test)).numpy()

    # ONNX
    onnx_pred = sess.run(None, {"X": X_test})[0]

    max_diff = np.max(np.abs(py_pred - onnx_pred))
    row_sums = onnx_pred.sum(axis=1)
    print(f"  ONNX validation: max_diff={max_diff:.8f} "
          f"[{'OK' if max_diff < 1e-4 else 'MISMATCH'}]")
    print(f"  Row sums: min={row_sums.min():.6f} max={row_sums.max():.6f}")


def export_all(net, scaler, columns: List[str], n_features: int,
               out_dir: str = None):
    """Export model, scaler, and columns to the ONNX directory."""
    out_dir = out_dir or ONNX_DIR
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, "mlp_fold_0.onnx")
    scaler_path = os.path.join(out_dir, "mlp_scaler_0.json")
    cols_path = os.path.join(out_dir, "mlp_cols.json")

    print("\nExporting model artifacts...")
    export_onnx(net, n_features, onnx_path)
    export_scaler(scaler, scaler_path)
    export_columns(columns, cols_path)
    validate_onnx(onnx_path, net, n_features)
    print(f"\nAll artifacts saved to: {out_dir}")
