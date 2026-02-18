import json, glob, os

# Find the latest results
dirs = sorted(glob.glob("data/cities/models_v5_6_sampler/*/sweep_results.json"))
if not dirs:
    print("No results found")
    exit()

path = dirs[-1]
print(f"Results from: {path}\n")
r = json.load(open(path))
ranked = sorted(r, key=lambda x: x["mean_r2"], reverse=True)

CLASS_NAMES = ["tree_cover", "grassland", "cropland", "built_up", "bare_sparse", "water"]

print(f"{'Rk':>2} {'Config':30s} {'R2':>7} {'tree':>7} {'grass':>7} {'crop':>7} {'built':>7} {'bare':>7} {'water':>7} {'Params':>10}")
print("-" * 120)
for i, c in enumerate(ranked, 1):
    pc = c["per_city"]["frankfurt"]["per_class_r2"]
    print(f"{i:2d} {c['name']:30s} {c['mean_r2']:>7.4f} "
          f"{pc['tree_cover']:>7.4f} {pc['grassland']:>7.4f} {pc['cropland']:>7.4f} "
          f"{pc['built_up']:>7.4f} {pc['bare_sparse']:>7.4f} {pc['water']:>7.4f} "
          f"{c['n_params']:>10,}")

# Also check old round 1 results
dirs_old = sorted(glob.glob("data/cities/models_v5_6_sampler/20260218_141527/sweep_results.json"))
if dirs_old:
    print(f"\n\n--- Round 1 results (from first v5.6 run) ---")
    r2 = json.load(open(dirs_old[0]))
    ranked2 = sorted(r2, key=lambda x: x["mean_r2"], reverse=True)
    for i, c in enumerate(ranked2, 1):
        pc = c["per_city"]["frankfurt"]["per_class_r2"]
        print(f"{i:2d} {c['name']:30s} {c['mean_r2']:>7.4f} "
              f"{pc['tree_cover']:>7.4f} {pc['grassland']:>7.4f} {pc['cropland']:>7.4f} "
              f"{pc['built_up']:>7.4f} {pc['bare_sparse']:>7.4f} {pc['water']:>7.4f} "
              f"{c['n_params']:>10,}")
