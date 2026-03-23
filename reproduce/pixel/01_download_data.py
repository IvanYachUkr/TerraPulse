#!/usr/bin/env python3
"""
Step 1: Download satellite data for the pixel-wise CatBoost classifier.

Same data as the MLP pipeline. If you've already run reproduce/mlp/01_download_data.py,
this step can be skipped — the data is shared.

Usage:
    python 01_download_data.py
    python 01_download_data.py --cities munich nuremberg
"""

import os, sys

# Reuse the MLP download script — same data needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLP_REPRODUCE = os.path.join(PROJECT_ROOT, "reproduce", "mlp")
sys.path.insert(0, MLP_REPRODUCE)

from importlib import import_module
step1 = import_module("01_download_data")

if __name__ == "__main__":
    step1.main()
