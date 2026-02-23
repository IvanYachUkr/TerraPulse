#!/usr/bin/env python3
"""
Cross-reference LUCAS LC1 codes with WorldCover labels at matching grid cells.
For each LUCAS point falling in our training/test tiles, find the cell it lands in
and compare the LUCAS label against the dominant WorldCover class.

Focus: ambiguous categories C32, H11, H12, and clean categories for baseline.
"""
import os, sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    CITIES, CLASS_NAMES, city_dir, city_anchor_path
)

LUCAS_DIR = os.path.join(PROJECT_ROOT, "data", "lucas")

# LUCAS code -> our 7 classes mapping (clean cases only for now)
CODE_TO_CLASS = {}
# A = built_up
for prefix in ["A1", "A2", "A3"]:
    CODE_TO_CLASS[prefix] = "built_up"
# B = cropland
for prefix in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
    CODE_TO_CLASS[prefix] = "cropland"
# C10-C22 = tree_cover (living forest)
for prefix in ["C10", "C21", "C22", "C23"]:
    CODE_TO_CLASS[prefix] = "tree_cover"
# C31-C33 = AMBIGUOUS (tracked separately)
# D = shrubland
for prefix in ["D10", "D20"]:
    CODE_TO_CLASS[prefix] = "shrubland"
# E = grassland
for prefix in ["E10", "E20", "E30"]:
    CODE_TO_CLASS[prefix] = "grassland"
# F = bare_sparse (including F30 lichen/moss for now)
for prefix in ["F10", "F20", "F30", "F40"]:
    CODE_TO_CLASS[prefix] = "bare_sparse"
# G = water
for prefix in ["G10", "G11", "G12", "G20", "G30", "G40", "G50"]:
    CODE_TO_CLASS[prefix] = "water"

def map_lc1_to_class(lc1):
    """Map LUCAS LC1 code to our class, or return the code itself for ambiguous."""
    if pd.isna(lc1):
        return None
    lc1 = str(lc1)
    # Try exact match first (3-char like C10), then 2-char prefix
    for length in [3, 2]:
        prefix = lc1[:length]
        if prefix in CODE_TO_CLASS:
            return CODE_TO_CLASS[prefix]
    return "UNMAPPED:" + lc1


def main():
    # Load LUCAS
    df = pd.read_csv(os.path.join(LUCAS_DIR, "EU_LUCAS_2022.csv"), low_memory=False)
    df = df[df["SURVEY_LC1"].notna()].copy()
    
    # Use GPS coords with fallback to theoretical
    df["lat"] = df["SURVEY_GPS_LAT"].fillna(df["POINT_LAT"])
    df["lon"] = df["SURVEY_GPS_LONG"].fillna(df["POINT_LONG"])
    
    # For each city, find LUCAS points that fall inside, then look up WC label
    results = []
    
    for city in CITIES:
        w, s, e, n = city.bbox
        mask = (df["lon"] >= w) & (df["lon"] <= e) & (df["lat"] >= s) & (df["lat"] <= n)
        pts = df[mask]
        if len(pts) == 0:
            continue
        
        # Load labels for this city
        label_path = os.path.join(city_dir(city), "labels_2021.parquet")
        if not os.path.exists(label_path):
            continue
        labels = pd.read_parquet(label_path)
        
        # Load anchor to get transform
        anchor_path = city_anchor_path(city)
        if not os.path.exists(anchor_path):
            continue
        with rasterio.open(anchor_path) as src:
            transform = src.transform
            width = src.width
            height = src.height
        
        # Grid dimensions (100m cells = 10 pixels per cell)
        n_cols = width // 10
        n_rows = height // 10
        
        for _, pt in pts.iterrows():
            # Convert lat/lon to pixel coords
            try:
                # First reproject lat/lon to the city's CRS
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:{}".format(city.epsg),
                                                    always_xy=True)
                x_proj, y_proj = transformer.transform(pt["lon"], pt["lat"])
                
                # Get pixel position
                col_px, row_px = ~transform * (x_proj, y_proj)
                
                # Convert to cell index
                cell_col = int(col_px) // 10
                cell_row = int(row_px) // 10
                
                if cell_col < 0 or cell_col >= n_cols or cell_row < 0 or cell_row >= n_rows:
                    continue
                
                cell_id = cell_row * n_cols + cell_col
                
                # Look up WC label
                cell_labels = labels[labels["cell_id"] == cell_id]
                if len(cell_labels) == 0:
                    continue
                
                cl = cell_labels.iloc[0]
                wc_fracs = {cn: cl[cn] for cn in CLASS_NAMES}
                wc_dominant = max(wc_fracs, key=wc_fracs.get)
                wc_dom_frac = wc_fracs[wc_dominant]
                
                lc1 = str(pt["SURVEY_LC1"])
                our_class = map_lc1_to_class(lc1)
                
                results.append({
                    "city": city.name,
                    "is_test": city.is_test,
                    "lc1": lc1,
                    "lc1_prefix": lc1[:3] if len(lc1) >= 3 else lc1[:2],
                    "lucas_class": our_class,
                    "wc_dominant": wc_dominant,
                    "wc_dom_frac": wc_dom_frac,
                    "agrees": our_class == wc_dominant if our_class and "UNMAPPED" not in str(our_class) else None,
                    **{("wc_" + cn): wc_fracs[cn] for cn in CLASS_NAMES},
                })
            except Exception as ex:
                continue
    
    rdf = pd.DataFrame(results)
    print("Total matched LUCAS-WC pairs: {:,}".format(len(rdf)))
    print()
    
    # === Overall agreement ===
    clean = rdf[rdf["agrees"].notna()]
    print("=== OVERALL AGREEMENT (clean categories) ===")
    print("Agreement: {:.1f}% ({:,} / {:,})".format(
        clean["agrees"].mean() * 100, int(clean["agrees"].sum()), len(clean)))
    print()
    
    # === Per-class agreement ===
    print("=== PER-CLASS AGREEMENT ===")
    for cls in CLASS_NAMES:
        sub = clean[clean["lucas_class"] == cls]
        if len(sub) == 0:
            continue
        agree = sub["agrees"].mean() * 100
        # What does WC think these are?
        wc_dist = sub["wc_dominant"].value_counts()
        top3 = "; ".join("{}: {:.0f}%".format(k, v/len(sub)*100) for k, v in wc_dist.head(3).items())
        print("  {:15s} agree={:5.1f}%  n={:>4}  WC says: {}".format(cls, agree, len(sub), top3))
    print()
    
    # === AMBIGUOUS: C32 (few trees) ===
    print("=== C32 (FEW SCATTERED TREES) ===")
    c32 = rdf[rdf["lc1"].str.startswith("C32", na=False)]
    if len(c32) > 0:
        wc_dist = c32["wc_dominant"].value_counts()
        for k, v in wc_dist.items():
            print("  WC says {:15s}: {:>3} ({:.0f}%)".format(k, v, v/len(c32)*100))
        tree_agree = (c32["wc_dominant"] == "tree_cover").mean() * 100
        print("  -> tree_cover agreement: {:.0f}%  (n={})".format(tree_agree, len(c32)))
    else:
        print("  No C32 points in our tiles")
    print()
    
    # === AMBIGUOUS: C31 (recently felled) ===
    print("=== C31 (RECENTLY FELLED) ===")
    c31 = rdf[rdf["lc1"].str.startswith("C31", na=False)]
    if len(c31) > 0:
        wc_dist = c31["wc_dominant"].value_counts()
        for k, v in wc_dist.items():
            print("  WC says {:15s}: {:>3} ({:.0f}%)".format(k, v, v/len(c31)*100))
    else:
        print("  No C31 points in our tiles")
    print()
    
    # === AMBIGUOUS: C33 (transitional) ===
    print("=== C33 (TRANSITIONAL WOODLAND) ===")
    c33 = rdf[rdf["lc1"].str.startswith("C33", na=False)]
    if len(c33) > 0:
        wc_dist = c33["wc_dominant"].value_counts()
        for k, v in wc_dist.items():
            print("  WC says {:15s}: {:>3} ({:.0f}%)".format(k, v, v/len(c33)*100))
    else:
        print("  No C33 points in our tiles")
    print()
    
    # === AMBIGUOUS: H11 (inland marsh) ===
    print("=== H11 (INLAND MARSH) ===")
    h11 = rdf[rdf["lc1"].str.startswith("H11", na=False)]
    if len(h11) > 0:
        wc_dist = h11["wc_dominant"].value_counts()
        for k, v in wc_dist.items():
            print("  WC says {:15s}: {:>3} ({:.0f}%)".format(k, v, v/len(h11)*100))
    else:
        print("  No H11 points in our tiles")
    print()
    
    # === AMBIGUOUS: H12 (peatbog) ===
    print("=== H12 (PEATBOG) ===")
    h12 = rdf[rdf["lc1"].str.startswith("H12", na=False)]
    if len(h12) > 0:
        wc_dist = h12["wc_dominant"].value_counts()
        for k, v in wc_dist.items():
            print("  WC says {:15s}: {:>3} ({:.0f}%)".format(k, v, v/len(h12)*100))
    else:
        print("  No H12 points in our tiles")
    print()
    
    # === AMBIGUOUS: F30 (lichen/moss) ===
    print("=== F30 (LICHEN/MOSS) ===")
    f30 = rdf[rdf["lc1"].str.startswith("F30", na=False)]
    if len(f30) > 0:
        wc_dist = f30["wc_dominant"].value_counts()
        for k, v in wc_dist.items():
            print("  WC says {:15s}: {:>3} ({:.0f}%)".format(k, v, v/len(f30)*100))
    else:
        print("  No F30 points in our tiles")
    
    # Save for later use
    rdf.to_csv(os.path.join(LUCAS_DIR, "lucas_wc_alignment.csv"), index=False)
    print("\nSaved alignment data to lucas_wc_alignment.csv")


if __name__ == "__main__":
    main()
