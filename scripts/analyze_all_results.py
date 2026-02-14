"""Quick analysis: compare all MLP + tree results from V4 onwards."""
import pandas as pd
import os

d = "reports/phase8/tables"
all_results = []

files = {
    "V4": "mlp_overnight_v4.csv",
    "V5": "mlp_v5_deep.csv",
    "V5.5": "mlp_v5_5_arch.csv",
    "V6": "mlp_v6_arch.csv",
    "V7": "mlp_v7_arch.csv",
    "V8_trees": "mlp_v8_trees.csv",
    "trees_old": "trees_reduced_features.csv",
}

for src, fname in files.items():
    p = os.path.join(d, fname)
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["source"] = src
        if "run_name" in df.columns and "name" not in df.columns:
            df["name"] = df["run_name"]
        all_results.append(df)

big = pd.concat(all_results, ignore_index=True)
valid = big[big["r2_uniform"].notna()].copy()

agg = valid.groupby(["name", "source"]).agg(
    r2_mean=("r2_uniform", "mean"),
    r2_std=("r2_uniform", "std"),
    mae_mean=("mae_mean_pp", "mean"),
    mae_std=("mae_mean_pp", "std"),
    folds=("fold", "count"),
).reset_index()

multi = agg[agg.folds >= 2].copy()

print("=" * 90)
print("TOP 20 BY R2 (min 2 folds)")
print("=" * 90)
for _, r in multi.sort_values("r2_mean", ascending=False).head(20).iterrows():
    rs = f"{r.r2_std:.4f}" if pd.notna(r.r2_std) else "  n/a"
    ms = f"{r.mae_std:.2f}" if pd.notna(r.mae_std) else " n/a"
    nm = r["name"]
    print(f"  {r.source:10s} R2={r.r2_mean:.4f}+-{rs}  MAE={r.mae_mean:.2f}+-{ms}pp  f={int(r.folds)}  {nm}")

print()
print("=" * 90)
print("TOP 20 BY MAE (min 2 folds)")
print("=" * 90)
for _, r in multi.sort_values("mae_mean").head(20).iterrows():
    rs = f"{r.r2_std:.4f}" if pd.notna(r.r2_std) else "  n/a"
    ms = f"{r.mae_std:.2f}" if pd.notna(r.mae_std) else " n/a"
    nm = r["name"]
    print(f"  {r.source:10s} MAE={r.mae_mean:.2f}+-{ms}pp  R2={r.r2_mean:.4f}+-{rs}  f={int(r.folds)}  {nm}")

print()
print("=" * 90)
print("BEST PER SWEEP (by R2)")
print("=" * 90)
for src in ["V4", "V5", "V5.5", "V6", "V7", "V8_trees", "trees_old"]:
    sub = multi[multi.source == src].sort_values("r2_mean", ascending=False)
    if len(sub):
        b = sub.iloc[0]
        rs = f"{b.r2_std:.4f}" if pd.notna(b.r2_std) else "n/a"
        nm = b["name"]
        print(f"  {src:10s} R2={b.r2_mean:.4f}+-{rs}  MAE={b.mae_mean:.2f}pp  f={int(b.folds)}  {nm}")

print()
print("=" * 90)
print("BEST PER SWEEP (by MAE)")
print("=" * 90)
for src in ["V4", "V5", "V5.5", "V6", "V7", "V8_trees", "trees_old"]:
    sub = multi[multi.source == src].sort_values("mae_mean")
    if len(sub):
        b = sub.iloc[0]
        rs = f"{b.r2_std:.4f}" if pd.notna(b.r2_std) else "n/a"
        nm = b["name"]
        print(f"  {src:10s} MAE={b.mae_mean:.2f}pp  R2={b.r2_mean:.4f}+-{rs}  f={int(b.folds)}  {nm}")
