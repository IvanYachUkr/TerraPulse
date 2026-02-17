"""
Data-driven training region selection.

Downloads ESA WorldCover 2020+2021 labels for ~15 candidate German cities,
computes diversity metrics, and ranks them to find the best 5 for training.

Metrics:
  - Shannon entropy of class proportions (higher = more diverse)
  - Number of non-trivial classes (proportion > 1%)
  - Change rate between 2020 and 2021 labels
  - Class balance (1 - Gini coefficient)

Usage:
    python scripts/select_training_regions.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Candidate cities: 15 diverse German cities covering all geographic regions
# bbox = [west, south, east, north], ~20-30 km regions
# ---------------------------------------------------------------------------
CANDIDATES = {
    # North / Coastal
    "Hamburg":     [9.80, 53.45, 10.15, 53.63],
    "Bremen":     [8.70, 53.00, 8.95, 53.14],
    "Rostock":    [12.05, 54.05, 12.25, 54.18],
    # West / Rhine
    "Cologne":    [6.85, 50.87, 7.10, 51.00],
    "Dusseldorf": [6.70, 51.17, 6.90, 51.30],
    "Essen":      [6.92, 51.40, 7.12, 51.52],
    # East
    "Leipzig":    [12.25, 51.27, 12.50, 51.40],
    "Dresden":    [13.65, 51.00, 13.90, 51.12],
    "Berlin":     [13.25, 52.45, 13.55, 52.58],
    # South
    "Munich":     [11.40, 48.05, 11.70, 48.22],
    "Stuttgart":  [9.08, 48.72, 9.28, 48.84],
    "Augsburg":   [10.82, 48.33, 11.00, 48.42],
    # Central
    "Frankfurt":  [8.55, 50.05, 8.80, 50.20],
    "Kassel":     [9.40, 51.27, 9.58, 51.36],
    "Hannover":   [9.65, 52.32, 9.85, 52.42],
}

# Nuremberg excluded intentionally (held-out test city)
NUREMBERG_BBOX = [10.95, 49.38, 11.20, 49.52]

# WorldCover class IDs and names
WC_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# Classes we care about (matching our model's 6 output classes)
TARGET_CLASSES = {10, 30, 40, 50, 60, 80}


def download_worldcover_tile(bbox, year=2021, cache_dir=None):
    """Download WorldCover for a bbox via Planetary Computer STAC."""
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from rasterio.windows import from_bounds

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "region_selection")
    os.makedirs(cache_dir, exist_ok=True)

    # Check cache
    cache_key = f"wc_{year}_{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    # Collection is 'esa-worldcover', items have year in their ID
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
    all_items = list(search.items())

    # Filter by year from item ID (e.g., "ESA_WorldCover_10m_2021_v200_N48E009")
    items = [it for it in all_items if f"_{year}_" in it.id]
    if not items:
        print(f"  WARNING: No WorldCover {year} items found for {bbox}")
        return None

    # Sign the item to get access
    item = pc.sign(items[0])
    asset = item.assets["map"]
    href = asset.href

    with rasterio.open(href) as src:
        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=window)

    np.save(cache_path, data)
    return data


def compute_metrics(data_2021, data_2020=None):
    """Compute diversity metrics for a WorldCover tile."""
    if data_2021 is None:
        return None

    flat = data_2021.ravel()
    total = len(flat)

    # Class proportions
    proportions = {}
    for cls_id, cls_name in WC_CLASSES.items():
        count = np.sum(flat == cls_id)
        proportions[cls_name] = count / total

    # Shannon entropy (higher = more diverse)
    p = np.array([v for v in proportions.values() if v > 0])
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(WC_CLASSES))
    normalized_entropy = entropy / max_entropy

    # Number of target classes with > 1% presence
    target_props = {k: v for k, v in proportions.items()
                    if any(cls_id for cls_id, name in WC_CLASSES.items()
                           if name == k and cls_id in TARGET_CLASSES)}
    n_classes_present = sum(1 for v in target_props.values() if v > 0.01)

    # Gini coefficient for balance
    sorted_p = np.sort(p)[::-1]
    n = len(sorted_p)
    if n > 1:
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * sorted_p) / (n * np.sum(sorted_p))) - (n + 1) / n
    else:
        gini = 0
    balance = 1 - gini

    # Change rate (if 2020 data available)
    change_rate = 0.0
    if data_2020 is not None:
        changed_pixels = np.sum(data_2021 != data_2020)
        change_rate = changed_pixels / total

    # Ensure we have built_up (essential for urban change detection)
    built_up_pct = proportions.get("built_up", 0)

    return {
        "entropy": normalized_entropy,
        "n_target_classes": n_classes_present,
        "balance": balance,
        "change_rate": change_rate,
        "built_up_pct": built_up_pct,
        "tree_pct": proportions.get("tree_cover", 0),
        "grass_pct": proportions.get("grassland", 0),
        "crop_pct": proportions.get("cropland", 0),
        "water_pct": proportions.get("water", 0),
        "bare_pct": proportions.get("bare_sparse", 0),
        "pixels": total,
    }


def composite_score(metrics):
    """Compute a composite diversity score."""
    return (
        0.35 * metrics["entropy"] +
        0.15 * (metrics["n_target_classes"] / 6.0) +
        0.15 * metrics["balance"] +
        0.15 * min(metrics["change_rate"] * 100, 1.0) +  # cap at 1%
        0.10 * min(metrics["built_up_pct"] * 5, 1.0) +   # want >20% built-up
        0.10 * (1 - abs(metrics["tree_pct"] - 0.3))       # prefer ~30% tree
    )


def main():
    print("=" * 70)
    print("TRAINING REGION SELECTION")
    print("Downloading WorldCover labels for 15 German cities...")
    print("=" * 70)

    results = {}

    for city, bbox in CANDIDATES.items():
        print(f"\n  Downloading {city}...")
        try:
            data_2021 = download_worldcover_tile(bbox, year=2021)
            data_2020 = download_worldcover_tile(bbox, year=2020)
            metrics = compute_metrics(data_2021, data_2020)
            if metrics:
                metrics["score"] = composite_score(metrics)
                results[city] = metrics
                print(f"    {data_2021.shape} pixels, entropy={metrics['entropy']:.3f}, "
                      f"classes={metrics['n_target_classes']}, "
                      f"change={metrics['change_rate']:.4f}, "
                      f"score={metrics['score']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Also analyze Nuremberg for comparison
    print(f"\n  Downloading Nuremberg (for comparison)...")
    try:
        data_2021 = download_worldcover_tile(NUREMBERG_BBOX, year=2021)
        data_2020 = download_worldcover_tile(NUREMBERG_BBOX, year=2020)
        metrics = compute_metrics(data_2021, data_2020)
        if metrics:
            metrics["score"] = composite_score(metrics)
            results["Nuremberg*"] = metrics
            print(f"    entropy={metrics['entropy']:.3f}, "
                  f"classes={metrics['n_target_classes']}, "
                  f"score={metrics['score']:.3f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Build rankings table
    print("\n" + "=" * 70)
    print("DIVERSITY RANKINGS")
    print("=" * 70)

    df = pd.DataFrame(results).T
    df = df.sort_values("score", ascending=False)

    # Print table
    header = (f"{'Rank':<5} {'City':<15} {'Score':<7} {'Entropy':<9} "
              f"{'Classes':<9} {'Change%':<9} {'Built%':<8} "
              f"{'Tree%':<8} {'Crop%':<8} {'Water%':<8}")
    print(header)
    print("-" * len(header))

    for i, (city, row) in enumerate(df.iterrows()):
        marker = " <-- TEST" if city == "Nuremberg*" else ""
        print(f"{i+1:<5} {city:<15} {row['score']:.3f}   "
              f"{row['entropy']:.3f}     "
              f"{int(row['n_target_classes']):<9} "
              f"{row['change_rate']*100:.2f}%    "
              f"{row['built_up_pct']*100:.1f}%    "
              f"{row['tree_pct']*100:.1f}%    "
              f"{row['crop_pct']*100:.1f}%    "
              f"{row['water_pct']*100:.1f}%{marker}")

    # Select top 5 (excluding Nuremberg)
    training_cities = [c for c in df.index if c != "Nuremberg*"][:5]

    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED TRAINING CITIES (top 5 by diversity):")
    print(f"{'=' * 70}")
    for i, city in enumerate(training_cities):
        bbox = CANDIDATES[city]
        print(f"  {i+1}. {city:<15} bbox={bbox}")

    print(f"\n  HELD-OUT TEST: Nuremberg  bbox={NUREMBERG_BBOX}")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "region_selection")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "training_cities": {city: CANDIDATES[city] for city in training_cities},
        "test_city": {"Nuremberg": NUREMBERG_BBOX},
        "rankings": {city: {k: float(v) for k, v in row.items()}
                     for city, row in df.iterrows()},
    }
    config_path = os.path.join(out_dir, "region_selection.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved to: {config_path}")

    # Save CSV
    csv_path = os.path.join(out_dir, "region_rankings.csv")
    df.to_csv(csv_path)
    print(f"  Rankings saved to: {csv_path}")


if __name__ == "__main__":
    main()
