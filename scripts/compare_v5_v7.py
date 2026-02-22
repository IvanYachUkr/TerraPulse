#!/usr/bin/env python3
"""
Compare V5 (old) vs V7 (new) features and composites for test cities.
Generates a comparison HTML page viewable in browser.
"""

import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "v7_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = ["bremen", "hamburg", "duesseldorf"]


def compare_features():
    """Compare feature parquets: NaN counts, value distributions."""
    print("=" * 60)
    print("FEATURE COMPARISON: V5 (old) vs V7 (new)")
    print("=" * 60)

    for city in CITIES:
        old_path = os.path.join(CITIES_DIR, city, "features",
                                "features_rust_2020_2021.parquet")
        new_path = os.path.join(CITIES_DIR, city, "features_v7",
                                "features_rust_2020_2021.parquet")

        if not os.path.exists(old_path):
            print(f"\n  [{city}] No old features — skip")
            continue
        if not os.path.exists(new_path):
            print(f"\n  [{city}] No new features — skip")
            continue

        old = pd.read_parquet(old_path)
        new = pd.read_parquet(new_path)

        # Exclude control columns
        control = {"cell_id", "valid_fraction", "low_valid_fraction",
                   "reflectance_scale", "full_features_computed"}
        old_feat = [c for c in old.columns if c not in control]
        new_feat = [c for c in new.columns if c not in control]

        old_vals = old[old_feat].values.astype(np.float32)
        new_vals = new[new_feat].values.astype(np.float32)

        old_nan = np.isnan(old_vals).sum()
        new_nan = np.isnan(new_vals).sum()
        old_zero = (old_vals == 0).sum()
        new_zero = (new_vals == 0).sum()

        print(f"\n  [{city}]")
        print(f"    Old: {old.shape[0]} cells, {len(old_feat)} features")
        print(f"    New: {new.shape[0]} cells, {len(new_feat)} features")
        print(f"    Old NaN: {old_nan:,} ({old_nan/old_vals.size*100:.3f}%)")
        print(f"    New NaN: {new_nan:,} ({new_nan/new_vals.size*100:.3f}%)")
        print(f"    Old zeros: {old_zero:,} ({old_zero/old_vals.size*100:.2f}%)")
        print(f"    New zeros: {new_zero:,} ({new_zero/new_vals.size*100:.2f}%)")

        # Compare shared columns
        shared = sorted(set(old_feat) & set(new_feat))
        if shared:
            old_shared = old[shared].values.astype(np.float32)
            new_shared = new[shared].values.astype(np.float32)
            # Replace NaN with 0 for comparison
            old_s = np.nan_to_num(old_shared)
            new_s = np.nan_to_num(new_shared)
            diff = np.abs(old_s - new_s)
            print(f"    Shared columns: {len(shared)}")
            print(f"    Mean abs diff: {diff.mean():.6f}")
            print(f"    Max abs diff:  {diff.max():.6f}")
            # Find columns with largest differences
            col_diffs = diff.mean(axis=0)
            top5 = np.argsort(col_diffs)[-5:][::-1]
            print(f"    Top differing columns:")
            for i in top5:
                print(f"      {shared[i]}: mean_diff={col_diffs[i]:.6f}")

        del old, new


def generate_composite_images():
    """Generate RGB comparison PNGs (downsampled) for browser viewing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("COMPOSITE IMAGE COMPARISON")
    print("=" * 60)

    import rasterio

    image_paths = []

    for city in CITIES:
        old_dir = os.path.join(CITIES_DIR, city, "raw")
        new_dir = os.path.join(CITIES_DIR, city, "raw_v7")

        for season in ["spring", "summer", "autumn"]:
            old_path = os.path.join(old_dir,
                f"sentinel2_{city}_2020_{season}.tif")
            new_path = os.path.join(new_dir,
                f"sentinel2_{city}_2020_{season}.tif")

            if not os.path.exists(old_path) or not os.path.exists(new_path):
                continue

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            for ax, path, label in [
                (axes[0], old_path, "V5 (Python median)"),
                (axes[1], new_path, "V7 (Rust Q1)"),
            ]:
                with rasterio.open(path) as src:
                    # Read RGB bands (B4=red, B3=green, B2=blue = bands 3,2,1 in 0-indexed)
                    r = src.read(3).astype(np.float32)  # B04
                    g = src.read(2).astype(np.float32)  # B03
                    b = src.read(1).astype(np.float32)  # B02

                # Replace NODATA
                for band in [r, g, b]:
                    band[band < -9000] = np.nan

                # Normalize to [0,1] for display
                # If data is in DN scale, divide by 10000
                stack = np.stack([r, g, b])
                p2 = np.nanpercentile(stack, 2)
                p98 = np.nanpercentile(stack, 98)
                if p98 > 1.5:  # DN scale
                    r = r / 10000
                    g = g / 10000
                    b = b / 10000

                # Stretch
                rgb = np.stack([r, g, b], axis=-1)
                lo, hi = np.nanpercentile(rgb, [2, 98])
                rgb = np.clip((rgb - lo) / (hi - lo + 1e-10), 0, 1)
                rgb = np.nan_to_num(rgb, nan=0.0)

                # Downsample for display
                step = max(1, rgb.shape[0] // 500)
                rgb_small = rgb[::step, ::step]

                nan_pct = np.isnan(r).sum() / r.size * 100
                ax.imshow(rgb_small)
                ax.set_title(f"{label}\n{season} 2020 — NaN: {nan_pct:.1f}%",
                            fontsize=12)
                ax.axis("off")

            fig.suptitle(f"{city.title()} — {season} 2020", fontsize=14, fontweight="bold")
            plt.tight_layout()
            out_path = os.path.join(OUT_DIR, f"{city}_{season}_comparison.png")
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            image_paths.append(out_path)
            print(f"  Saved: {os.path.basename(out_path)}")

    return image_paths


def generate_html(image_paths):
    """Generate HTML comparison page."""
    html = """<!DOCTYPE html>
<html><head>
<title>V5 vs V7 Composite Comparison</title>
<style>
body { background: #1a1a2e; color: #eee; font-family: sans-serif; padding: 20px; }
h1 { text-align: center; color: #e94560; }
h2 { color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 4px; }
img { max-width: 100%; border: 2px solid #333; border-radius: 8px; margin: 10px 0; }
.card { background: #16213e; padding: 16px; border-radius: 8px; margin: 16px 0; }
</style>
</head><body>
<h1>V5 (Python median) vs V7 (Rust Q1) Composite Comparison</h1>
"""
    for path in image_paths:
        name = os.path.basename(path).replace("_comparison.png", "").replace("_", " ").title()
        html += f'<div class="card"><h2>{name}</h2>\n'
        html += f'<img src="{os.path.basename(path)}" alt="{name}"></div>\n'

    html += "</body></html>"

    html_path = os.path.join(OUT_DIR, "comparison.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\n  HTML: {html_path}")
    return html_path


if __name__ == "__main__":
    compare_features()
    paths = generate_composite_images()
    html_path = generate_html(paths)
    print(f"\nOpen in browser: file:///{html_path.replace(os.sep, '/')}")
