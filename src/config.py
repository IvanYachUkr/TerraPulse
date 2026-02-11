"""
Centralized configuration loader for the TerraPulse data pipeline.

Usage:
    from src.config import CFG, PROJECT_ROOT

    bbox = CFG["aoi"]["bbox"]
    bands = CFG["sentinel2"]["bands"]
"""

import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "data_config.yml")

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# Convenience paths
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_V2_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "v2")
LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_V2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "v2")
GRID_DIR = os.path.join(PROJECT_ROOT, "data", "grid")
OSM_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "osm")
GRID_REF_PATH = os.path.join(PROJECT_ROOT, CFG["grid"]["reference_file"])
