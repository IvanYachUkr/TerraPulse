"""Aggregate all MLP results V5-V16 across 5-fold CV and rank them."""
import pandas as pd
import glob
import os
import re

base = r"c:\Users\vanya\Documents\ML_1_sem\final\reports\phase8\tables"

frames = []
for f in sorted(glob.glob(os.path.join(base, "mlp_v*.csv"))):
    bn = os.path.basename(f)
    if "summary" in bn or "v8" in bn:
        continue
    m = re.search(r"mlp_v(\d+\w*)", bn)
    ver = m.group(1) if m else bn
    df = pd.read_csv(f)
    if "r2_uniform" not in df.columns or "fold" not in df.columns:
        continue
    name_col = "name" if "name" in df.columns else ("texture_combo" if "texture_combo" in df.columns else None)
    if not name_col:
        continue
    rec = df[[name_col, "fold", "r2_uniform", "mae_mean_pp"]].copy()
    rec = rec.rename(columns={name_col: "cfg_name"})
    rec["version"] = "V" + ver
    rec["n_features"] = df.get("n_features", df.get("n_feat", pd.Series([0]*len(df))))
    rec["feature_set"] = df.get("feature_set", df.get("texture_combo", pd.Series([""]*len(df))))
    rec["data_source"] = df.get("data_source", pd.Series(["python_parquet"]*len(df)))
    rec["model_head"] = df.get("head", pd.Series(["ilr"]*len(df)))
    frames.append(rec)

all_folds = pd.concat(frames, ignore_index=True)

g = all_folds.groupby(["version", "cfg_name"]).agg(
    r2_mean=("r2_uniform", "mean"),
    r2_std=("r2_uniform", "std"),
    mae_mean=("mae_mean_pp", "mean"),
    n_folds=("fold", "nunique"),
    n_features=("n_features", "first"),
    feature_set=("feature_set", "first"),
    data_source=("data_source", "first"),
    model_head=("model_head", "first"),
).reset_index()

five = g[g["n_folds"] == 5].sort_values("r2_mean", ascending=False).reset_index(drop=True)

print("Total 5-fold configs:", len(five))
print("Versions:", sorted(five.version.unique()))
print()

for i in range(min(10, len(five))):
    r = five.iloc[i]
    nf = int(r.n_features) if pd.notna(r.n_features) else 0
    folds = all_folds[(all_folds.version == r.version) & (all_folds.cfg_name == r.cfg_name)].sort_values("fold")
    fold_vals = []
    for _, x in folds.iterrows():
        fold_vals.append("F{}={:.3f}".format(int(x.fold), x.r2_uniform))
    fold_str = "  ".join(fold_vals)

    rank = i + 1
    print("#{:2d}  [{}]".format(rank, r.version))
    print("    R2 = {:.4f} +/- {:.4f}   MAE = {:.2f} pp".format(r.r2_mean, r.r2_std, r.mae_mean))
    print("    Config:     {}".format(r.cfg_name))
    print("    Features:   {} ({})".format(r.feature_set, nf))
    print("    Head:       {}".format(r.model_head))
    print("    Data:       {}".format(r.data_source))
    print("    Per-fold:   {}".format(fold_str))
    print()
