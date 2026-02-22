#!/usr/bin/env python3
"""
V8 MLP Sweep — identical to V7/V6 but uses V8 features (97 training cities,
Rust-downloaded, spatial NN NaN fill). Monkey-patches city_features_dir to
point to features_v8/ instead of features/.

After the sweep completes, automatically exports the best model to ONNX
for the Rust inference pipeline.

Usage:
    .venv/Scripts/python.exe scripts/sweep_mlp_v8.py
    .venv/Scripts/python.exe scripts/sweep_mlp_v8.py --export-only
"""

import os, sys, json, pickle, shutil

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")

# ---------------------------------------------------------------------------
# Monkey-patch city_features_dir BEFORE importing sweep_mlp_v6
# ---------------------------------------------------------------------------
import scripts.run_multi_city_pipeline_v5 as v5_mod

_original_city_features_dir = v5_mod.city_features_dir


def _city_features_dir_v8(city):
    """Override: point to features_v7/ (reuse V7 data, no duplication)."""
    return os.path.join(v5_mod.city_dir(city), "features_v7")


# Patch the module-level function so sweep_mlp_v6 picks it up
v5_mod.city_features_dir = _city_features_dir_v8

V8_SWEEP_DIR = os.path.join(CITIES_DIR, "models_v8_sweep")

# Also patch the output directory
os.environ["SWEEP_OUTPUT_DIR"] = V8_SWEEP_DIR


# ---------------------------------------------------------------------------
# ONNX Export: architecture classes (must match sweep_mlp_v6.py)
# ---------------------------------------------------------------------------
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
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)


class ConstantMLP(nn.Module):
    def __init__(self, in_features, n_classes, n_layers, width, dropout=0.15,
                 input_dropout=0.05):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = []
        prev_dim = in_features
        for _ in range(n_layers):
            layers.append(PlainBlock(prev_dim, width, dropout))
            prev_dim = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)


def export_best_onnx(sweep_dir):
    """Find the best config from sweep results and export to ONNX."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("ONNX EXPORT: Best V8 Model")
    print(sep)

    # Find the most recent run dir (timestamped)
    run_dirs = sorted([
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d))
        and os.path.exists(os.path.join(sweep_dir, d, "sweep_results.json"))
    ])
    if not run_dirs:
        # Results might be in sweep_dir itself
        if os.path.exists(os.path.join(sweep_dir, "sweep_results.json")):
            run_dir = sweep_dir
        else:
            print("  ERROR: No sweep_results.json found!")
            return
    else:
        run_dir = os.path.join(sweep_dir, run_dirs[-1])

    # Load results
    with open(os.path.join(run_dir, "sweep_results.json")) as f:
        results = json.load(f)

    if not results:
        print("  ERROR: Empty sweep results!")
        return

    # Find best by mean_r2
    best = max(results, key=lambda r: r.get("mean_r2", -999))
    print(f"  Best config: {best['name']}")
    print(f"  Mean R2:     {best['mean_r2']:.4f}")
    print(f"  Params:      {best['n_params']:,}")
    print(f"  Widths:      {best['widths']}")
    print(f"  Type:        {best['type']}")

    # Load columns
    cols_path = os.path.join(run_dir, "mlp_cols.json")
    with open(cols_path) as f:
        mlp_cols = json.load(f)
    n_features = len(mlp_cols)
    n_classes = 7

    # Rebuild model architecture
    widths = best["widths"]
    dropout = best.get("dropout", 0.15)

    if best["type"] == "T":
        net = TaperedMLP(n_features, n_classes, widths, dropout=dropout,
                         input_dropout=0.05)
    else:
        net = ConstantMLP(n_features, n_classes, len(widths), widths[0],
                          dropout=dropout, input_dropout=0.05)

    # Load weights
    pt_path = os.path.join(run_dir, f"{best['name']}.pt")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    print(f"  Loaded weights from {pt_path}")

    # ONNX output dir
    onnx_dir = os.path.join(PROJECT_ROOT, "data", "pipeline_output", "models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Export ONNX
    print(f"\n  Exporting to ONNX...")
    onnx_path = os.path.join(onnx_dir, "mlp_fold_0.onnx")
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
        print(f"  Validation: max_diff={max_diff:.8f} "
              f"[{'OK' if max_diff < 1e-4 else 'MISMATCH'}]")
        print(f"  Row sums: [{row_sums.min():.6f}, {row_sums.max():.6f}] "
              f"(should be ~1.0)")
    except ImportError:
        print("  WARNING: onnxruntime not available, skipping validation")

    # Export scaler as JSON
    print(f"\n  Exporting scaler...")
    pkl_path = os.path.join(run_dir, "scaler.pkl")
    with open(pkl_path, "rb") as f:
        scaler = pickle.load(f)

    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    json_path = os.path.join(onnx_dir, "mlp_scaler_0.json")
    with open(json_path, "w") as f:
        json.dump(scaler_data, f)
    print(f"  Scaler: {scaler.n_features_in_} features → {json_path}")

    # Export column list
    with open(os.path.join(onnx_dir, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)
    print(f"  Columns: {len(mlp_cols)} → mlp_cols.json")

    # Summary
    print(f"\n{sep}")
    print("ONNX export complete. Files:")
    for name in sorted(os.listdir(onnx_dir)):
        size = os.path.getsize(os.path.join(onnx_dir, name))
        print(f"  {name:40s} {size/1024:>8.0f} KB")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true",
                        help="Skip sweep, just export best model to ONNX")
    args, remaining = parser.parse_known_args()

    if args.export_only:
        export_best_onnx(V8_SWEEP_DIR)
        sys.exit(0)

    # Run V6 sweep (with monkey-patched features dir)
    print(f"\nRunning V8 MLP Sweep (features from features_v8/)")
    print(f"Output dir: {V8_SWEEP_DIR}\n")

    sys.argv = [sys.argv[0]] + remaining  # pass through extra args
    exec(open(os.path.join(PROJECT_ROOT, "scripts", "sweep_mlp_v6.py")).read())

    # After sweep completes, auto-export best model to ONNX
    export_best_onnx(V8_SWEEP_DIR)
