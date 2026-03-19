#!/usr/bin/env python3
"""
BOHB Hyperparameter Sweep for MLP.

Separated from the main training pipeline — only needed for initial
hyperparameter exploration, unlikely to be rerun frequently.

Usage:
    .venv\\Scripts\\python -u mlp/sweep.py
    .venv\\Scripts\\python -u mlp/sweep.py --max-trials 5 --max-budget 20  # dry run
"""

import gc
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

# Fix numpy serialization for Pyro4 RPC
import serpent

def _numpy_serializer(obj, serpent_serializer, outputstream, indentlevel):
    if isinstance(obj, np.integer):
        serpent_serializer._serialize(int(obj), outputstream, indentlevel)
    elif isinstance(obj, np.floating):
        serpent_serializer._serialize(float(obj), outputstream, indentlevel)
    elif isinstance(obj, np.bool_):
        serpent_serializer._serialize(bool(obj), outputstream, indentlevel)
    elif isinstance(obj, (np.str_, np.bytes_)):
        serpent_serializer._serialize(str(obj), outputstream, indentlevel)
    elif isinstance(obj, np.ndarray):
        serpent_serializer._serialize(obj.tolist(), outputstream, indentlevel)
    else:
        serpent_serializer._serialize(str(obj), outputstream, indentlevel)

for np_type in [np.int32, np.int64, np.float32, np.float64,
                np.bool_, np.str_, np.bytes_, np.ndarray, np.intc, np.intp]:
    try:
        serpent.register_class(np_type, _numpy_serializer)
    except Exception:
        pass

from mlp.config import (
    ALL_TRAIN, ALL_TEST, VAL_CITY_NAMES, EXCLUDED_CITY_NAMES,
    CITIES_DIR, N_CLASSES, SEED,
)
from mlp.features import select_features, get_common_columns, city_has_sar
from mlp.model import TaperedMLP
from mlp.data import (
    build_memmap_dataset, fit_scaler, apply_scaler_inplace, load_val_to_gpu,
)
from mlp.train import normalize_targets, train_one_config, compute_objective

# ---------------------------------------------------------------------------
# Architecture search space
# ---------------------------------------------------------------------------
BASE_ARCHS = {
    "T_512_256_128_64":   [512, 256, 128, 64],
    "T_1024_512_256_64":  [1024, 512, 256, 64],
    "T_2048_512_128":     [2048, 512, 128],
    "T_2048_1024_512":    [2048, 1024, 512],
}


def get_configspace():
    cs = CS.ConfigurationSpace(seed=SEED)
    cs.add([
        CSH.CategoricalHyperparameter("arch", choices=list(BASE_ARCHS.keys())),
        CSH.UniformFloatHyperparameter("dropout", 0.05, 0.35, default_value=0.15),
        CSH.UniformFloatHyperparameter("input_dropout", 0.0, 0.15, default_value=0.05),
        CSH.UniformFloatHyperparameter("lr", 1e-4, 5e-3, default_value=1e-3, log=True),
        CSH.UniformFloatHyperparameter("weight_decay", 1e-5, 1e-2, default_value=1e-4, log=True),
        CSH.UniformFloatHyperparameter("mixup_alpha", 0.0, 0.5, default_value=0.3),
        CSH.UniformFloatHyperparameter("label_threshold", 0.0, 0.10, default_value=0.0),
        CSH.UniformFloatHyperparameter("mixup_prob", 0.3, 0.8, default_value=0.5),
        CSH.CategoricalHyperparameter("activation", choices=["silu", "gelu", "mish"],
                                      default_value="silu"),
        CSH.CategoricalHyperparameter("batch_size", choices=[2048, 4096],
                                      default_value=4096),
    ])
    return cs


# ---------------------------------------------------------------------------
# BOHB Worker
# ---------------------------------------------------------------------------
class MLPWorker(Worker):
    def __init__(self, X_train, y_norm, val_tensors, n_features, val_name,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trial_count = 0
        self.X_train = X_train
        self.y_norm = y_norm
        self.val_tensors = val_tensors
        self.n_features = n_features
        self.val_name = val_name

    def compute(self, config, budget, **kwargs):
        self.trial_count += 1
        budget = int(budget)
        arch_name = config["arch"]
        widths = BASE_ARCHS[arch_name]

        print(f"\n{'='*60}")
        print(f"  Trial {self.trial_count} | {arch_name} | {budget} epochs")
        print(f"{'='*60}")

        net = TaperedMLP(
            self.n_features, N_CLASSES, widths,
            dropout=config["dropout"],
            activation=config["activation"],
            input_dropout=config["input_dropout"],
        ).to(self.device)

        train_config = {
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
            "max_epochs": budget,
            "mixup_alpha": config["mixup_alpha"],
            "mixup_prob": config["mixup_prob"],
            "label_threshold": config["label_threshold"],
        }

        results = train_one_config(
            net, self.X_train, self.y_norm, self.val_tensors,
            train_config, self.device, val_city_name=self.val_name,
        )

        # Save checkpoint
        trial_dir = os.path.join(output_dir, "trials")
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(
            net.state_dict(),
            os.path.join(trial_dir, f"trial_{self.trial_count}_{arch_name}.pt"),
        )

        # Log
        trial_result = {
            "trial": self.trial_count,
            "arch": arch_name,
            "widths": widths,
            "config": {k: (int(v) if isinstance(v, np.integer) else
                          float(v) if isinstance(v, np.floating) else
                          str(v) if isinstance(v, (np.str_, np.bytes_)) else v)
                      for k, v in dict(config).items()},
            "budget": budget,
            **results,
        }
        log_path = os.path.join(output_dir, "trial_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(trial_result) + "\n")

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {"loss": -results["combined"], "info": trial_result}


# Module-level output_dir set in main()
output_dir = None


def main():
    import argparse
    global output_dir

    parser = argparse.ArgumentParser(description="BOHB MLP Sweep")
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--min-budget", type=int, default=15)
    parser.add_argument("--max-budget", type=int, default=300)
    parser.add_argument("--eta", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(CITIES_DIR, "models_mlp_sweep")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  BOHB MLP Sweep")
    print(f"{'='*70}")

    # ---- Data pipeline (same as run.py) ----
    all_sar = [c for c in ALL_TRAIN + ALL_TEST if city_has_sar(c)]
    train_cities = [c for c in all_sar
                    if c.name not in VAL_CITY_NAMES
                    and c.name not in EXCLUDED_CITY_NAMES]
    val_cities = [c for c in all_sar if c.name in VAL_CITY_NAMES]

    all_columns = get_common_columns(train_cities + val_cities)
    mlp_cols = select_features(all_columns)
    n_features = len(mlp_cols)
    print(f"  Features: {n_features}")

    mmap_dir = os.path.join(output_dir, "_mmap_cache")
    X_path, y_path, n_samples = build_memmap_dataset(
        train_cities, mlp_cols, mmap_dir,
    )

    X_train = np.memmap(X_path, dtype=np.float32, mode="r+",
                        shape=(n_samples, n_features))
    y_train = np.memmap(y_path, dtype=np.float32, mode="r+",
                        shape=(n_samples, N_CLASSES))

    scaler = fit_scaler(X_train)
    apply_scaler_inplace(X_train, scaler)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "mlp_cols.json"), "w") as f:
        json.dump(mlp_cols, f)

    val_tensors = load_val_to_gpu(val_cities, mlp_cols, scaler, device)
    y_norm = normalize_targets(np.array(y_train))

    val_name = "berlin" if "berlin" in val_tensors else list(val_tensors.keys())[0]
    gc.collect()

    # ---- BOHB ----
    logging.getLogger("hpbandster").setLevel(logging.WARNING)

    NS = hpns.NameServer(run_id="mlp_sweep", host="127.0.0.1", port=None)
    NS.start()

    worker = MLPWorker(
        X_train, y_norm, val_tensors, n_features, val_name,
        nameserver="127.0.0.1", nameserver_port=NS.port,
        run_id="mlp_sweep",
    )
    worker.run(background=True)

    bohb = BOHB(
        configspace=get_configspace(),
        run_id="mlp_sweep",
        nameserver="127.0.0.1", nameserver_port=NS.port,
        min_budget=args.min_budget, max_budget=args.max_budget,
        eta=args.eta,
    )

    result = bohb.run(n_iterations=args.max_trials)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # ---- Report ----
    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    best_config = id2config[incumbent]["config"]
    best_runs = result.get_runs_by_id(incumbent)
    best_loss = min(r.loss for r in best_runs)

    print(f"\n{'='*70}")
    print(f"  BOHB COMPLETE -- best combined: {-best_loss:.4f}")
    print(f"{'='*70}")
    for k, v in sorted(best_config.items()):
        print(f"    {k:20s}: {v}")

    with open(os.path.join(output_dir, "best_config.json"), "w") as f:
        json.dump({"config": best_config, "loss": float(best_loss)}, f, indent=2)


if __name__ == "__main__":
    main()
