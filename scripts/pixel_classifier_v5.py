"""V5: Train CatBoost with V1-style RANDOM sampling (not stratified).
Loads V2 features but samples 100K pixels randomly per city (2x V4) to boost accuracy.
Applies tuned hyperparameter configs to match/beat LightGBM."""
import gc, json, os, sys, time, warnings
import numpy as np
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pixel_classifier_v2 import (
    SEED, N_CLASSES, CLASS_NAMES, CITIES_DIR,
    build_pixel_features, ts,
)
from scripts.run_multi_city_pipeline_v5 import CITIES

EXCLUDED = {"nuremberg"}
VAL_CITIES = {
    "finnish_lakeland", "danish_farmland", "tabernas_desert",
    "sardinia_maquis", "crete_phrygana", "iceland_highlands",
    "lapland_tundra", "ireland_bog_pasture", "hortobagy_puszta",
    "vojvodina_cropland", "camargue_wetland", "pyrenees_meadows",
    "munich", "seville", "stockholm",
}
# Massively increase sample size for V5 to boost accuracy compared to V4
MAX_PX = 150000

CACHE_PATH = os.path.join(CITIES_DIR, 'pixel_cache_v5', f'data_random{MAX_PX//1000}k.npz')
NAMES_PATH = os.path.join(CITIES_DIR, 'pixel_cache_v5', 'names.json')


def load_cities_random(cities, max_px):
    """Load V2 features with RANDOM (not stratified) subsampling."""
    rng = np.random.RandomState(SEED)
    all_X, all_y = [], []
    feat_names = None

    for i, city in enumerate(cities):
        print(f"\n  [{ts()}] [{i+1}/{len(cities)}] {city.name}...")
        result = build_pixel_features(city)
        if result[0] is None:
            continue
        X, y, H, W, names = result

        if feat_names is None:
            feat_names = names

        # RANDOM sampling - just like V1
        if len(X) > max_px:
            idx = rng.choice(len(X), max_px, replace=False)
            X = X[idx]
            y = y[idx]
            print(f"    Random sampled: {len(X):,} pixels")
        else:
            print(f"    Kept all: {len(X):,} pixels")

        all_X.append(X.astype(np.float16))
        all_y.append(y)
        del X, y; gc.collect()

    return all_X, all_y, feat_names


def main():
    np.random.seed(SEED)
    out_dir = os.path.join(CITIES_DIR, "models_pixel_v5")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  V5: CatBoost + RANDOM sampling (100K max/city)")
    print(f"{'='*70}\n")

    # Check cache
    if os.path.exists(CACHE_PATH):
        print(f"[{ts()}] Loading cached random-sampled data...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        X_train = data['X_train'].astype(np.float32)
        y_train = data['y_train']
        X_val = data['X_val'].astype(np.float32)
        y_val = data['y_val']
        with open(NAMES_PATH) as f:
            feat_names = json.load(f)
        print(f"  Loaded in cache: train={X_train.shape}, val={X_val.shape}")
    else:
        train_cities = [c for c in CITIES
                        if c.name not in EXCLUDED and c.name not in VAL_CITIES]
        val_cities = [c for c in CITIES if c.name in VAL_CITIES]

        print(f"[{ts()}] Loading TRAIN ({len(train_cities)} cities) with RANDOM sampling...")
        all_X_train, all_y_train, feat_names = load_cities_random(train_cities, MAX_PX)

        print(f"\n[{ts()}] Loading VAL ({len(val_cities)} cities) with RANDOM sampling...")
        all_X_val, all_y_val, _ = load_cities_random(val_cities, MAX_PX)

        X_train = np.concatenate(all_X_train).astype(np.float32)
        y_train = np.concatenate(all_y_train)
        X_val = np.concatenate(all_X_val).astype(np.float32)
        y_val = np.concatenate(all_y_val)
        del all_X_train, all_y_train, all_X_val, all_y_val; gc.collect()

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez_compressed(CACHE_PATH, X_train=X_train.astype(np.float16),
                            y_train=y_train, X_val=X_val.astype(np.float16),
                            y_val=y_val)
        with open(NAMES_PATH, 'w') as f:
            json.dump(feat_names, f)
        print(f"\n[{ts()}] Cached to {CACHE_PATH}")

    print(f"\n  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    print(f"  Val:   {X_val.shape[0]:,} x {X_val.shape[1]}")
    for sn, sy in [("Train", y_train), ("Val", y_val)]:
        cls, cnt = np.unique(sy, return_counts=True); tot = cnt.sum()
        print(f"\n  {sn} distribution:")
        for c, n in zip(cls, cnt):
            print(f"    {CLASS_NAMES[c]:>15}: {n:>8,} ({100*n/tot:5.1f}%)")

    # Inverse frequency class weights (like V4)
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    class_weights_inv = {int(c): total / (len(classes) * cnt)
                         for c, cnt in zip(classes, counts)}

    # Strong configurations to beat LGBM
    configs = [
        # 1: Strong depth-8, with class weights
        {'name': 'deep_weighted', 'depth': 8, 'trees': 3000, 'lr': 0.03, 'l2': 3.0, 'cw': class_weights_inv},
        # 2: Strong depth-8, NO class weights (trusting random sample priors often boosts overall accuracy)
        {'name': 'deep_unweighted', 'depth': 8, 'trees': 3000, 'lr': 0.03, 'l2': 3.0, 'cw': None},
        # 3: Low LR + more trees (better generalisation), unweighted
        {'name': 'lowlr_unweighted', 'depth': 8, 'trees': 4000, 'lr': 0.02, 'l2': 5.0, 'cw': None},
        # 4: Fast baseline (depth 6)
        {'name': 'fast_baseline', 'depth': 6, 'trees': 2000, 'lr': 0.05, 'l2': 2.0, 'cw': class_weights_inv},
    ]

    from sklearn.metrics import accuracy_score, classification_report

    results = {}
    for config in configs:
        cname = config['name']
        print(f"\n{'='*70}")
        print(f"  CatBoost [{cname}] — RANDOM sampling {MAX_PX//1000}K")
        print(f"{'='*70}")

        # Try GPU first
        for dev in ['GPU', 'CPU']:
            try:
                params = {
                    'iterations': config['trees'],
                    'depth': config['depth'],
                    'learning_rate': config['lr'],
                    'l2_leaf_reg': config['l2'],
                    'random_seed': SEED,
                    'task_type': dev,
                    'loss_function': 'MultiClass',
                    'eval_metric': 'MultiClass',
                    'verbose': 100,
                    'early_stopping_rounds': 80,
                    'use_best_model': True,
                }
                if config['cw'] is not None:
                    params['class_weights'] = config['cw']
                if dev == 'GPU':
                    params['devices'] = '0'
                    params['gpu_ram_part'] = 0.95

                print(f"\n  [{ts()}] Training on {dev}...")
                model = CatBoostClassifier(**params)
                train_pool = Pool(X_train, y_train, feature_names=feat_names)
                val_pool = Pool(X_val, y_val, feature_names=feat_names)
                model.fit(train_pool, eval_set=val_pool)
                print(f"  Best iteration: {model.get_best_iteration()}")
                break
            except Exception as e:
                print(f"  {dev} failed: {e}")
                if dev == 'CPU':
                    raise

        # Evaluate
        pred = model.predict(X_val).flatten().astype(int)
        acc = accuracy_score(y_val, pred)
        report = classification_report(y_val, pred, target_names=CLASS_NAMES, digits=4)
        print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(report)
        results[f'catboost_{cname}'] = {'accuracy': float(acc), 'report': report}

        # Save
        path = os.path.join(out_dir, f"catboost_pixel_v5_{cname}.cbm")
        model.save_model(path)
        print(f"  Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    with open(os.path.join(out_dir, "metrics_pixel_v5.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[{ts()}] Done! V5 models saved to {out_dir}")


if __name__ == "__main__":
    main()
