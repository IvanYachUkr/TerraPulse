import json, glob, os
path = glob.glob("data/cities/models_v5_sweep/*/sweep_results.json")[0]
data = json.load(open(path))
ranked = sorted(data, key=lambda x: -x["mean_r2"])
print(f"Completed: {len(data)}/30 configs\n")
fmt = "{:28s} {:>10s} {:>8s} {:>8s} {:>4s} {:>6s} {:>7s} {:>7s} {:>7s}"
print(fmt.format("Config","Params","MeanR2","MAE","Ep","Time","Nur","Fra","Muc"))
print("-"*95)
for r in ranked:
    n = r["per_city"]["nuremberg"]["r2"]
    f = r["per_city"]["frankfurt"]["r2"]
    m = r["per_city"]["munich"]["r2"]
    print(f"{r['name']:28s} {r['n_params']:>10,} {r['mean_r2']:>8.4f} {r['mean_mae_pp']:>7.2f}pp {r['best_epoch']:>4d} {r['time_s']:>5.0f}s {n:>7.4f} {f:>7.4f} {m:>7.4f}")
