#!/usr/bin/env python3
"""
precompute_change_metrics.py – Compute change-specific metrics from
Nuremberg prediction bins (2018-2025) and label bins (2020, 2021).

Outputs: src/dashboard/data/nuremberg_dashboard/change_metrics.json

Metrics computed:
  1. Pixel stability across all years
  2. Annual change rate per consecutive year pair
  3. False change rate (predicted vs label change for 2020→2021)
  4. Transition matrix (top class transitions)
  5. Per-class stability
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_dashboard"

ANCHOR_H = 2850
ANCHOR_W = 2550
YEARS = list(range(2018, 2026))  # 2018-2025

CLASS_ORDER = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]
N_CLASSES = len(CLASS_ORDER)


def load_bin(name):
    """Load a res1 bin file as (H, W) uint8 array."""
    path = BIN_DIR / name
    if not path.exists():
        return None
    return np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(ANCHOR_H, ANCHOR_W)


def main():
    # Load all prediction bins
    preds = {}
    for y in YEARS:
        arr = load_bin(f"nuremberg_pred_{y}_res1.bin")
        if arr is not None:
            preds[y] = arr
            print(f"  Loaded predictions {y}: {(arr != 255).sum():,} valid pixels")
        else:
            print(f"  WARNING: Missing predictions for {y}")

    available_years = sorted(preds.keys())
    print(f"\n  Available years: {available_years}")

    # Load label bins
    labels = {}
    for y in [2020, 2021]:
        arr = load_bin(f"nuremberg_labels_{y}_res1.bin")
        if arr is not None:
            labels[y] = arr
            print(f"  Loaded labels {y}: {(arr != 255).sum():,} valid pixels")

    # Common valid mask (valid in all years)
    valid = np.ones((ANCHOR_H, ANCHOR_W), dtype=bool)
    for y in available_years:
        valid &= (preds[y] != 255) & (preds[y] < N_CLASSES)
    n_valid = valid.sum()
    print(f"\n  Common valid pixels across all years: {n_valid:,}")

    # ── 1. Pixel Stability ──
    # A pixel is "stable" if it has the same class in ALL years
    first = preds[available_years[0]][valid]
    stable_mask = np.ones(n_valid, dtype=bool)
    for y in available_years[1:]:
        stable_mask &= (preds[y][valid] == first)
    pixel_stability = float(stable_mask.sum() / n_valid)
    print(f"\n  Pixel stability (same class all years): {pixel_stability*100:.1f}%")

    # ── 2. Annual Change Rate ──
    annual_changes = {}
    for i in range(len(available_years) - 1):
        y1, y2 = available_years[i], available_years[i + 1]
        changed = (preds[y1][valid] != preds[y2][valid]).sum()
        rate = float(changed / n_valid)
        annual_changes[f"{y1}_{y2}"] = {
            "from": y1, "to": y2,
            "changed_pixels": int(changed),
            "total_pixels": int(n_valid),
            "change_rate": round(rate, 4),
        }
        print(f"  Change rate {y1}→{y2}: {rate*100:.2f}% ({changed:,} pixels)")

    # ── 3. False Change Rate (2020→2021, predictions vs labels) ──
    false_change_metrics = None
    if 2020 in labels and 2021 in labels and 2020 in preds and 2021 in preds:
        # Valid in both predictions AND labels
        lbl_valid = valid & (labels[2020] != 255) & (labels[2020] < N_CLASSES) \
                         & (labels[2021] != 255) & (labels[2021] < N_CLASSES)
        n_lbl = lbl_valid.sum()

        pred_changed = preds[2020][lbl_valid] != preds[2021][lbl_valid]
        label_changed = labels[2020][lbl_valid] != labels[2021][lbl_valid]

        # True positives: both predict and label say change
        tp = (pred_changed & label_changed).sum()
        # False positives: pred says change, label says no change
        fp = (pred_changed & ~label_changed).sum()
        # False negatives: pred says no change, label says change  
        fn = (~pred_changed & label_changed).sum()
        # True negatives: both say no change
        tn = (~pred_changed & ~label_changed).sum()

        false_change_rate = float(fp / max(pred_changed.sum(), 1))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))

        false_change_metrics = {
            "year_pair": "2020_2021",
            "total_pixels": int(n_lbl),
            "pred_changes": int(pred_changed.sum()),
            "label_changes": int(label_changed.sum()),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "false_change_rate": round(false_change_rate, 4),
            "change_precision": round(precision, 4),
            "change_recall": round(recall, 4),
        }
        print(f"\n  False change rate (2020→2021): {false_change_rate*100:.1f}%")
        print(f"  Change precision: {precision*100:.1f}%, recall: {recall*100:.1f}%")

    # ── 4. Transition Matrix (aggregate across all consecutive years) ──
    transition_counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for i in range(len(available_years) - 1):
        y1, y2 = available_years[i], available_years[i + 1]
        a = preds[y1][valid]
        b = preds[y2][valid]
        changed = a != b
        for from_cls in range(N_CLASSES):
            for to_cls in range(N_CLASSES):
                if from_cls == to_cls:
                    continue
                transition_counts[from_cls, to_cls] += ((a == from_cls) & (b == to_cls) & changed).sum()

    # Top 10 transitions
    flat_idx = np.argsort(transition_counts.ravel())[::-1]
    top_transitions = []
    for idx in flat_idx[:10]:
        from_cls = int(idx // N_CLASSES)
        to_cls = int(idx % N_CLASSES)
        count = int(transition_counts[from_cls, to_cls])
        if count == 0:
            break
        top_transitions.append({
            "from": CLASS_ORDER[from_cls],
            "to": CLASS_ORDER[to_cls],
            "count": count,
            "pct": round(count / max(transition_counts.sum(), 1) * 100, 2),
        })
    print(f"\n  Top transitions:")
    for t in top_transitions[:5]:
        print(f"    {t['from']} → {t['to']}: {t['count']:,} ({t['pct']}%)")

    # ── 5. Per-Class Stability ──
    per_class_stability = {}
    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        # For pixels classified as this class in the first year,
        # how many remained this class in all subsequent years?
        is_cls_first = preds[available_years[0]][valid] == cls_idx
        n_cls = is_cls_first.sum()
        if n_cls == 0:
            per_class_stability[cls_name] = {"count": 0, "stable_pct": 0.0}
            continue
        remained = is_cls_first.copy()
        for y in available_years[1:]:
            remained &= (preds[y][valid] == cls_idx)
        stability = float(remained.sum() / n_cls)
        per_class_stability[cls_name] = {
            "count": int(n_cls),
            "stable_pct": round(stability * 100, 1),
        }
        print(f"  {cls_name}: {stability*100:.1f}% stable ({n_cls:,} pixels)")

    # ── Build output JSON ──
    output = {
        "years": available_years,
        "total_valid_pixels": int(n_valid),
        "pixel_stability_pct": round(pixel_stability * 100, 1),
        "annual_changes": annual_changes,
        "false_change": false_change_metrics,
        "top_transitions": top_transitions,
        "per_class_stability": per_class_stability,
    }

    out_path = BIN_DIR / "change_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
