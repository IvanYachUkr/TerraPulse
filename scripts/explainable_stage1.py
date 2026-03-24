#!/usr/bin/env python3
"""
explainable_stage1.py – Linear/Logistic Regression for Urban Change Detection.

This script provides a highly explainable alternative to the Stage 1 Random Forest.
By using Logistic Regression with standardized features, we can directly inspect 
the coefficients to understand which spectral bands or socioeconomic stats 
drive the change likelihood prediction.

Metrics:
<<<<<<< HEAD
- Balanced Accuracy, F1, Precision, Recall.
- False Change Rate (FP on stable pixels).
=======
- Accuracy, F1, Precision, Recall, False Change Rate both on a balanced ant the whole testset
>>>>>>> 8d1ac27 (explainable model added to dashboard)
- Feature Importances (Normalized Coefficients).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# --- Paths & Grid ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
S2_DIR = DATA_DIR / "sentinel2"
WC_DIR = DATA_DIR / "worldcover"
STATS_DIR = DATA_DIR / "nuremberg_stats"

ANCHOR_CRS = "EPSG:32632"
ANCHOR_TRANSFORM = Affine(10.0, 0.0, 641740.0, 0.0, -10.0, 5492260.0)
ANCHOR_W, ANCHOR_H = 2550, 2850

WC_TO_CLASS = {10: 0, 20: 1, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 1}

def remap_wc(arr):
    out = np.full_like(arr, 255, dtype=np.uint8)
    for k, v in WC_TO_CLASS.items(): out[arr == k] = v
    return out

def reproject_to_grid(path, categorical=False):
    with rasterio.open(path) as src:
        dtype = np.uint8 if categorical else np.float32
        resampling = Resampling.nearest if categorical else Resampling.bilinear
        dst = np.full((src.count, ANCHOR_H, ANCHOR_W), 0 if categorical else np.nan, dtype=dtype)
        for i in range(src.count):
            reproject(rasterio.band(src, i+1), dst[i], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ANCHOR_TRANSFORM, dst_crs=ANCHOR_CRS, resampling=resampling)
        return dst if not categorical else dst[0]

def main():
    print("--- Explainable Stage 1 Change Detector ---")
    
    # 1. Load Boundary
    geo_p = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_boundary.geojson"
    from rasterio.features import rasterize
    gdf = gpd.read_file(geo_p).to_crs(ANCHOR_CRS)
    mask = rasterize([(g, 1) for g in gdf.geometry], out_shape=(ANCHOR_H, ANCHOR_W), transform=ANCHOR_TRANSFORM) == 1
    
    # 2. Load Features (2020)
    print("Loading 2020 features...")
    s2_20 = reproject_to_grid(S2_DIR / "s2_median_2020.tif")
    stats = reproject_to_grid(STATS_DIR / "rasterized_stats.tif")
    std_p = S2_DIR / "s2_ndvi_std_2020.tif"
    std = reproject_to_grid(std_p) if std_p.exists() else np.zeros((1, ANCHOR_H, ANCHOR_W))
    wc_20 = remap_wc(reproject_to_grid(WC_DIR / "worldcover_2020.tif", True))
    wc_21 = remap_wc(reproject_to_grid(WC_DIR / "worldcover_2021.tif", True))

    valid = (mask & np.all(np.isfinite(s2_20), axis=0) & np.all(np.isfinite(stats), axis=0))
    
    # Feature Engineering
    features = []
    for i in range(10): features.append(s2_20[i][valid]) # Bands
    b4, b8 = s2_20[2][valid], s2_20[6][valid]
    features.append((b8 - b4) / (b8 + b4 + 1e-8))      # NDVI
    features.append(wc_20[valid])                      # Prev LC
    for i in range(4): features.append(stats[i][valid]) # Stats
    features.append(std[0][valid])                     # Variability
    
    X = np.column_stack(features)
    y = (wc_21[valid] != wc_20[valid]).astype(int)
    
    feat_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", 
                  "NDVI", "Prev_LC", "Pop", "Housing", "Comm", "Cars", "NDVI_Std"]

    # 3. Balanced Dataset Split
    print("Balancing training set...")
    chg_idx = np.where(y == 1)[0]
    sta_idx = np.where(y == 0)[0]
    np.random.seed(42)
    sta_sample = np.random.choice(sta_idx, size=len(chg_idx), replace=False)
    
    bal_mask = np.concatenate([chg_idx, sta_sample])
    X_bal, y_bal = X[bal_mask], y[bal_mask]

    # 4. Standardize & Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)
    
    print("Training Optuna-optimized Logistic Regression...")
    model = LogisticRegression(
        C=0.0003022200373124438,
        penalty='l2',
        class_weight=None,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_scaled, y_bal)
    
    # 5. Metrics
    y_pred_bal = model.predict(X_scaled)
    X_full_scaled = scaler.transform(X)
    y_pred_full = model.predict(X_full_scaled)
    
    def get_metrics(y_true, y_p):
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        return {
            "Acc": accuracy_score(y_true, y_p),
            "F1": f1_score(y_true, y_p),
            "Prec": precision_score(y_true, y_p),
            "Rec": recall_score(y_true, y_p),
            "FCR": fp / (fp + tn) if (fp + tn) > 0 else 0
        }

    m_bal = get_metrics(y_bal, y_pred_bal)
    m_full = get_metrics(y, y_pred_full)
    
    print("\n--- Model Performance Comparison ---")
    print(f"{'Metric':<15} | {'Balanced Set':<15} | {'Whole Area (Imbalanced)':<25}")
    print("-" * 65)
    for k in ["Acc", "F1", "Prec", "Rec", "FCR"]:
        label = {"Acc":"Accuracy","F1":"F1-Score","Prec":"Precision","Rec":"Recall","FCR":"False Change Rate"}[k]
        val_bal = f"{m_bal[k]:.4f}"
        val_full = f"{m_full[k]:.4%}" if k == "FCR" else f"{m_full[k]:.4f}"
        print(f"{label:<15} | {val_bal:<15} | {val_full:<25}")

    # 6. Feature Importance
    print("\n--- Feature Importances (Logistic Weights) ---")
    weights = model.coef_[0]
    importance_df = pd.DataFrame({'Feature': feat_names, 'Weight': weights})
    importance_df['AbsWeight'] = importance_df['Weight'].abs()
    importance_df = importance_df.sort_values(by='AbsWeight', ascending=False)
    
    for _, row in importance_df.iterrows():
        sign = "(+)" if row['Weight'] > 0 else "(-)"
        print(f"{row['Feature']:10}: {row['Weight']:+.4f} {sign}")

    print("\nInterpretation:")
    print("- Positive weight: Increase in feature increases change probability.")
    print("- Magnitude: Larger absolute weights indicate higher importance.")

    # 7. Model Persistence
    import joblib
    MODEL_PATH = PROJECT_DIR / "models" / "explainable_stage1.joblib"
    joblib.dump({'model': model, 'scaler': scaler, 'features': feat_names}, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # 8. Precalculate Heatmap (for dashboard)
    print("\n--- Generating Dashboard Heatmap (2021) ---")
    s2_21 = reproject_to_grid(S2_DIR / "s2_median_2021.tif")
    if s2_21 is not None:
        valid_21 = (mask & np.all(np.isfinite(s2_21), axis=0) & np.all(np.isfinite(stats), axis=0))
        
        feat_21 = []
        for i in range(10): feat_21.append(s2_21[i][valid_21])
        b4_21, b8_21 = s2_21[2][valid_21], s2_21[6][valid_21]
        feat_21.append((b8_21 - b4_21) / (b8_21 + b4_21) + 1e-8)
        feat_21.append(wc_20[valid_21])
        for i in range(4): feat_21.append(stats[i][valid_21])
        feat_21.append(std[0][valid_21])
        
        X_21 = np.column_stack(feat_21)
        probs = model.predict_proba(scaler.transform(X_21))[:, 1]
        
        # Dashboard Export Prefix
        DASHBOARD_DATA = PROJECT_DIR / "src" / "dashboard" / "data" / "nuremberg_dashboard"
        flat_idx = np.where(valid_21.flatten())[0]
        
        h_base = np.full((ANCHOR_H, ANCHOR_W), 255, dtype=np.uint8)
        h_base.flat[flat_idx] = np.clip(probs * 254, 0, 254).astype(np.uint8)
        
        for r in range(1, 11):
            if r == 1: out_h = h_base
            else:
                rh, rw = ANCHOR_H // r, ANCHOR_W // r
                out_h = np.full((rh, rw), 255, dtype=np.uint8)
                for row in range(rh):
                    for col in range(rw):
                        block = h_base[row*r:(row+1)*r, col*r:(col+1)*r]
                        v = block[block != 255]
                        if v.size: out_h[row, col] = int(v.mean())
            
            p = DASHBOARD_DATA / f"explainable_heatmap_2021_res{r}.bin"
            p.write_bytes(out_h.tobytes())
            
        print(f"Precalculated 10 Explainable Heatmap resolutions (2021) saved to DASHBOARD_DATA.")

        # 9. Save Metrics for Dashboard
        import json
        metrics_p = DASHBOARD_DATA / "explainable_metrics.json"
        
        # Format similar to experimental_metrics.json
        dashboard_metrics = {
            "overall_accuracy": m_full["Acc"],
            "macro_f1": m_full["F1"],
            "recall": m_full["Rec"],
            "precision": m_full["Prec"],
            "false_change_rate": m_full["FCR"],
            "model_info": "Explainable Logistic Regression (optimized)",
            "training_info": "Balanced samples (change/no-change)",
            "features": feat_names,
            "balanced_metrics": m_bal
        }
        
        with open(metrics_p, 'w') as f:
            json.dump(dashboard_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_p}")
if __name__ == "__main__":
    main()
