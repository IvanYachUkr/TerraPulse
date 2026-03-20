"""
Precalculate experimental heatmap and changes binary files.

Compares labels_2020 vs experimental_pred_2021 at each resolution to produce:
- experimental_heatmap_res{N}.bin  — 0 (no change) / 255 (change predicted), boundary=255
- experimental_changes_res{N}.bin  — class index (0-5) for changed pixels, 254=no change, 255=boundary

Usage:
    python -m src.dashboard.precalculate_experimental
"""

import os
import numpy as np

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "data", "nuremberg_dashboard")
RESOLUTIONS = range(1, 11)


def precalculate():
    for res in RESOLUTIONS:
        labels_path = os.path.join(DASHBOARD_DIR, f"nuremberg_labels_2020_res{res}.bin")
        pred_path = os.path.join(DASHBOARD_DIR, f"experimental_pred_2021_res{res}.bin")

        if not os.path.exists(labels_path):
            print(f"  ⚠ Skipping res{res}: labels file not found")
            continue
        if not os.path.exists(pred_path):
            print(f"  ⚠ Skipping res{res}: experimental prediction file not found")
            continue

        labels = np.fromfile(labels_path, dtype=np.uint8)
        preds = np.fromfile(pred_path, dtype=np.uint8)

        if labels.shape != preds.shape:
            print(f"  ⚠ Skipping res{res}: shape mismatch {labels.shape} vs {preds.shape}")
            continue

        # Boundary mask: 255 in either source
        boundary = (labels == 255) | (preds == 255)
        changed = (labels != preds) & ~boundary

        # --- Heatmap: 0=no change, 255=change, boundary stays 255 ---
        heatmap = np.zeros_like(labels)
        heatmap[changed] = 255
        heatmap[boundary] = 255  # keep boundary as 255 (transparent in frontend)
        # Actually we need to distinguish boundary from "change".
        # Use: 0=no change, 200=change, 255=boundary
        # But the frontend expects 0-255 as a gradient...
        # Simpler: just use 0 vs 255 for changed, and 255 for boundary.
        # The frontend heatmap renderer treats 255 as boundary (transparent).
        # So we need a different value for "change". Let's use 254 for max change.
        heatmap = np.zeros_like(labels)
        heatmap[changed] = 254  # high likelihood
        heatmap[boundary] = 255  # boundary → transparent

        heatmap_path = os.path.join(DASHBOARD_DIR, f"experimental_heatmap_res{res}.bin")
        heatmap.tofile(heatmap_path)
        n_changed = int(changed.sum())
        total_valid = int((~boundary).sum())
        print(f"  ✓ res{res}: heatmap saved ({n_changed}/{total_valid} changed pixels, "
              f"{n_changed/max(total_valid,1)*100:.1f}%)")

        # --- Changes: class index for changed pixels, 254=no change, 255=boundary ---
        changes = np.full_like(labels, 254)  # default: no change
        changes[changed] = preds[changed]     # changed pixels → predicted class
        changes[boundary] = 255               # boundary → transparent

        changes_path = os.path.join(DASHBOARD_DIR, f"experimental_changes_res{res}.bin")
        changes.tofile(changes_path)
        print(f"           changes saved")

    print("\nDone! All experimental files generated.")


if __name__ == "__main__":
    print("Precalculating experimental heatmap & changes...")
    precalculate()
