#!/usr/bin/env python3
"""
Auto-launcher: polls for search completion, then starts CV with
stratified config selection (top 20 global + top 10 per feature set).

Usage:
    python scripts/launch_cv_after_search.py [--poll-interval 600] [--top-global 20] [--top-per-fs 10]

Runs until the search CSV stops growing, then:
1. Selects top-20 global + top-10 per feature_set (deduplicated)
2. Writes the selected config names to a JSON file
3. Launches run_mlp_overnight_v4.py --stage cv with custom config list
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEARCH_CSV = ROOT / "reports" / "phase8" / "tables" / "mlp_overnight_v4_search.csv"
SELECTION_JSON = ROOT / "reports" / "phase8" / "tables" / "cv_selected_configs.json"
V4_SCRIPT = ROOT / "scripts" / "run_mlp_overnight_v4.py"

# Cross-platform Python path
_venv_scripts = ROOT / ".venv" / "Scripts" / "python.exe"   # Windows
_venv_bin = ROOT / ".venv" / "bin" / "python"               # Linux/Mac
PYTHON = _venv_scripts if _venv_scripts.exists() else _venv_bin


def log(msg: str = ""):
    """Print with immediate flush for real-time output."""
    print(msg, flush=True)


def read_row_count() -> int:
    """Read current number of rows in search CSV (retry-safe)."""
    if not SEARCH_CSV.exists():
        return 0
    for attempt in range(3):
        try:
            df = pd.read_csv(SEARCH_CSV)
            return len(df)
        except Exception as e:
            if attempt < 2:
                time.sleep(1)  # wait for partial write to finish
            else:
                log(f"  WARNING: CSV read failed after 3 retries: {e}")
                return -1  # signal "could not read"
    return 0


def select_configs(top_global: int = 20, top_per_fs: int = 10) -> list[str]:
    """Select top-N global + top-N per feature_set, deduplicated."""
    if not SEARCH_CSV.exists():
        log(f"ERROR: Search CSV not found: {SEARCH_CSV}")
        sys.exit(1)

    df = pd.read_csv(SEARCH_CSV)

    # Drop error rows if present
    if "error" in df.columns:
        df = df[df["error"].isna()]

    # Only fold-0 results
    if "fold" in df.columns:
        df = df[df["fold"] == 0]

    df = df.dropna(subset=["r2_uniform"])

    if len(df) == 0:
        log("ERROR: No valid results in search CSV (all errors or empty)")
        sys.exit(1)

    # Top N global
    global_top = set(df.nlargest(top_global, "r2_uniform")["name"].tolist())

    # Top N per feature set (if column exists)
    per_fs_top = set()
    n_sets = 0
    if "feature_set" in df.columns:
        for fs, grp in df.groupby("feature_set"):
            per_fs_top.update(grp.nlargest(top_per_fs, "r2_uniform")["name"].tolist())
        n_sets = df["feature_set"].nunique()
    else:
        log("WARNING: 'feature_set' column not found — using global top only")

    combined = global_top | per_fs_top
    overlap = len(global_top & per_fs_top)

    log(f"\n{'='*60}")
    log("CONFIG SELECTION SUMMARY")
    log(f"{'='*60}")
    log(f"  Top {top_global} global:           {len(global_top)} configs")
    log(f"  Top {top_per_fs} per feature set:  {len(per_fs_top)} configs "
        f"({n_sets} sets)")
    log(f"  Overlap:                  {overlap}")
    log(f"  Unique total:             {len(combined)}")
    log(f"  CV runs:                  {len(combined)} x 5 folds = {len(combined)*5}")
    log(f"  Est. time:                {len(combined)*5*120/3600:.1f}h (at ~2min/model)")
    log(f"{'='*60}")

    # Show per-feature-set breakdown
    if "feature_set" in df.columns:
        log("\nPer feature set:")
        for fs in sorted(df["feature_set"].unique()):
            grp = df[df["feature_set"] == fs]
            top_names = set(grp.nlargest(top_per_fs, "r2_uniform")["name"])
            n_in = len(top_names & combined)
            best = grp["r2_uniform"].max()
            log(f"  {fs:30s}  best R2={best:.4f}  ({n_in} selected)")

    return sorted(combined)


def main():
    parser = argparse.ArgumentParser(description="Auto-launch CV after search completes")
    parser.add_argument("--poll-interval", type=int, default=600,
                        help="Seconds between poll checks (default: 600 = 10 min)")
    parser.add_argument("--top-global", type=int, default=20,
                        help="Top N configs globally (default: 20)")
    parser.add_argument("--top-per-fs", type=int, default=10,
                        help="Top N configs per feature set (default: 10)")
    parser.add_argument("--total-expected", type=int, default=1584,
                        help="Expected total search configs (default: 1584)")
    parser.add_argument("--max-epochs", type=int, default=300,
                        help="Max epochs for CV stage (default: 300)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select configs but don't launch CV")
    args = parser.parse_args()

    log(f"Auto-CV launcher started at {time.strftime('%H:%M:%S')}")
    log(f"  Poll interval: {args.poll_interval}s")
    log(f"  Selection: top {args.top_global} global + top {args.top_per_fs} per feature set")
    log(f"  Total expected: {args.total_expected}")
    log(f"  Search CSV: {SEARCH_CSV}")
    log(f"  Python: {PYTHON}")
    log()

    # ---- Phase 1: Poll until search finishes ----
    MIN_ROWS = 100                # don't declare done if barely started
    CONSECUTIVE_NEEDED = 2        # require 2 unchanged polls in a row
    prev_count = -1
    unchanged_streak = 0

    while True:
        count = read_row_count()
        ts = time.strftime("%H:%M:%S")

        # Skip this poll if CSV read failed
        if count < 0:
            log(f"[{ts}] CSV read error — skipping this poll")
            time.sleep(args.poll_interval)
            continue

        # Check if we hit the expected total
        if count >= args.total_expected:
            log(f"\n[{ts}] Search COMPLETE — {count}/{args.total_expected} rows (all done)")
            break

        # Check for stalled growth (search process finished)
        if count > MIN_ROWS and count == prev_count:
            unchanged_streak += 1
            if unchanged_streak >= CONSECUTIVE_NEEDED:
                log(f"\n[{ts}] Search COMPLETE — CSV unchanged for "
                    f"{unchanged_streak} polls at {count} rows")
                break
            log(f"[{ts}] {count}/{args.total_expected} "
                f"({100*count/args.total_expected:.0f}%) "
                f"— unchanged ({unchanged_streak}/{CONSECUTIVE_NEEDED})")
        else:
            unchanged_streak = 0
            log(f"[{ts}] {count}/{args.total_expected} "
                f"({100*count/args.total_expected:.0f}%) — in progress")

        prev_count = count
        time.sleep(args.poll_interval)

    # ---- Phase 2: Select configs ----
    selected = select_configs(args.top_global, args.top_per_fs)

    # Save selection for reproducibility
    SELECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_JSON, "w") as f:
        json.dump({
            "top_global": args.top_global,
            "top_per_fs": args.top_per_fs,
            "n_configs": len(selected),
            "search_csv_rows": read_row_count(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configs": selected,
        }, f, indent=2)
    log(f"\nSelection saved to {SELECTION_JSON}")

    if args.dry_run:
        log("\n--dry-run: skipping CV launch")
        return

    # ---- Phase 3: Launch CV ----
    log(f"\nLaunching CV stage with {len(selected)} configs...")
    cmd = [
        str(PYTHON), str(V4_SCRIPT),
        "--stage", "cv",
        "--max-epochs", str(args.max_epochs),
        "--cv-config-json", str(SELECTION_JSON),
    ]
    log(f"  Command: {' '.join(cmd)}")
    log(f"  Started at: {time.strftime('%H:%M:%S')}")
    log("=" * 60)

    # Use subprocess.call — os.execv doesn't work cleanly on Windows
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
