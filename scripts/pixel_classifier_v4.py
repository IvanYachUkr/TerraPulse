"""V4: Train CatBoost with V1-style RANDOM sampling (not stratified).
Loads V2 features (217) but samples 50K pixels randomly per city."""
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
MAX_PX = 50000

CACHE_PATH = os.path.join(CITIES_DIR, 'pixel_cache_v4', 'data_random50k.npz')
NAMES_PATH = os.path.join(CITIES_DIR, 'pixel_cache_v4', 'names.json')


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
    out_dir = os.path.join(CITIES_DIR, "models_pixel_v4")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  V4: CatBoost + RANDOM sampling (V1-style)")
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

    # Train CatBoost default config
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    class_weights = {int(c): total / (len(classes) * cnt)
                     for c, cnt in zip(classes, counts)}

    configs = [
        {'name': 'default', 'depth': 8, 'trees': 2000, 'lr': 0.05},
        {'name': 'shallow_fast', 'depth': 6, 'trees': 1500, 'lr': 0.08},
    ]

    from sklearn.metrics import accuracy_score, classification_report

    results = {}
    for config in configs:
        cname = config['name']
        print(f"\n{'='*70}")
        print(f"  CatBoost [{cname}] — RANDOM sampling")
        print(f"{'='*70}")

        # Try GPU first
        for dev in ['GPU', 'CPU']:
            try:
                params = {
                    'iterations': config['trees'],
                    'depth': config['depth'],
                    'learning_rate': config['lr'],
                    'l2_leaf_reg': 3.0,
                    'random_seed': SEED,
                    'task_type': dev,
                    'loss_function': 'MultiClass',
                    'eval_metric': 'MultiClass',
                    'class_weights': class_weights,
                    'verbose': 100,
                    'early_stopping_rounds': 80,
                    'use_best_model': True,
                }
                if dev == 'GPU':
                    params['devices'] = '0'

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
        path = os.path.join(out_dir, f"catboost_pixel_v4_{cname}.cbm")
        model.save_model(path)
        print(f"  Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    with open(os.path.join(out_dir, "metrics_pixel_v4.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[{ts()}] Done! V4 models saved to {out_dir}")


if __name__ == "__main__":
    main()
