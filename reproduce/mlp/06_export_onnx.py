#!/usr/bin/env python3
"""
Step 6: Export Model #7 to ONNX for Rust inference.

Exports the trained model to ONNX format and saves scaler parameters
as JSON for the terrapulse Rust predict pipeline.

Usage:
    python 06_export_onnx.py
"""

import json, os, pickle, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
V10_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")
ONNX_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline_output", "models", "onnx")

N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]


# Model #7 config
MODEL7 = {
    "arch": "T_1024_512_256_64",
    "widths": [1024, 512, 256, 64],
    "activation": "gelu",
    "dropout": 0.3255,
    "input_dropout": 0.0031,
    "trial": 77,
    "label_threshold": 0.021,
}


# ---------------------------------------------------------------------------
# Model architecture (must match training exactly)
# ---------------------------------------------------------------------------
def _make_norm(nt, d):
    if nt == "batchnorm": return nn.BatchNorm1d(d)
    return nn.Identity()


class PlainBlock(nn.Module):
    def __init__(self, din, dout, do=0.15, act="gelu", nt="batchnorm"):
        super().__init__()
        self.linear = nn.Linear(din, dout)
        self.norm = _make_norm(nt, dout)
        self.dropout = nn.Dropout(do)
        self.act_fn = {"gelu": lambda x: F.gelu(x, approximate="tanh"),
                       "silu": F.silu, "relu": F.relu, "mish": F.mish}[act]

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class TaperedMLP(nn.Module):
    def __init__(self, in_f, n_c, widths, do=0.15, act="gelu", ido=0.05):
        super().__init__()
        self.input_drop = nn.Dropout(ido) if ido > 0 else nn.Identity()
        layers, prev = [], in_f
        for w in widths:
            layers.append(PlainBlock(prev, w, do, act, "batchnorm"))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_c)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def export_forward(self, x):
        """Softmax output for ONNX (not log-softmax)."""
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad(): return self.forward(x).exp()


def main():
    print(f"\n{'='*70}")
    print(f"  Export Model #7 to ONNX")
    print(f"{'='*70}\n")

    # Load feature column list
    cols_path = os.path.join(V10_DIR, "mlp_cols.json")
    scaler_path = os.path.join(V10_DIR, "scaler.pkl")
    model_path = os.path.join(V10_DIR, f"trial_{MODEL7['trial']}_{MODEL7['arch']}.pt")

    for p, label in [(cols_path, "columns"), (scaler_path, "scaler"), (model_path, "model")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            print("Run 03_train_bohb_sweep.py or 04_train_model7.py first.")
            sys.exit(1)

    with open(cols_path) as f:
        mlp_cols = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    n_features = len(mlp_cols)
    print(f"  Features: {n_features}")

    # Build model and load weights
    net = TaperedMLP(n_features, N_CLASSES, MODEL7["widths"],
                     MODEL7["dropout"], MODEL7["activation"], MODEL7["input_dropout"])
    net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    net.eval()
    print(f"  Model loaded: {sum(p.numel() for p in net.parameters()):,} params")

    # Export ONNX
    os.makedirs(ONNX_DIR, exist_ok=True)
    onnx_path = os.path.join(ONNX_DIR, "mlp_fold_0.onnx")

    original_forward = net.forward
    net.forward = net.export_forward

    dummy = torch.randn(1, n_features)
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["X"],
        output_names=["probabilities"],
        dynamic_axes={"X": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    net.forward = original_forward
    print(f"  ONNX saved: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    # Save scaler as JSON for Rust
    scaler_json_path = os.path.join(ONNX_DIR, "mlp_scaler_0.json")
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    with open(scaler_json_path, "w") as f:
        json.dump(scaler_data, f)
    print(f"  Scaler saved: {scaler_json_path}")

    # Save column list
    cols_out = os.path.join(ONNX_DIR, "mlp_cols.json")
    with open(cols_out, "w") as f:
        json.dump(mlp_cols, f)
    print(f"  Columns saved: {cols_out}")

    # Save model config
    config_path = os.path.join(ONNX_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_type": "mlp_v10_bohb",
            "trial": MODEL7["trial"],
            "arch": MODEL7["arch"],
            "widths": MODEL7["widths"],
            "activation": MODEL7["activation"],
            "label_threshold": MODEL7["label_threshold"],
            "n_features": n_features,
            "n_classes": N_CLASSES,
            "class_names": CLASS_NAMES,
        }, f, indent=2)
    print(f"  Config saved: {config_path}")

    # Validate ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        rng = np.random.RandomState(42)
        X_test = rng.randn(100, n_features).astype(np.float32)

        net.eval()
        with torch.no_grad():
            py_pred = net.export_forward(torch.tensor(X_test)).numpy()
        onnx_pred = sess.run(None, {"X": X_test})[0]
        max_diff = np.max(np.abs(py_pred - onnx_pred))
        print(f"\n  ONNX validation: max_diff={max_diff:.8f} "
              f"[{'OK' if max_diff < 1e-4 else 'MISMATCH'}]")
    except ImportError:
        print("\n  Note: pip install onnxruntime for ONNX validation")

    print(f"\n  All artifacts saved to: {ONNX_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
