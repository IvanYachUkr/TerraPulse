# 📋 Mapping Urban Change in Nuremberg — Full Project Checklist (SOLO)

> **Course**: Machine Learning WT 25/26 — Grabocka, Asano, Frey — UTN  
> **Objective**: Build a tabular ML system that predicts land-cover composition and change in Nuremberg, deliver an interactive product with uncertainty communication.  
> **Mode**: **Solo** (ignore team assignment fields)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]`  | Not started |
| `[/]`  | In progress |
| `[x]`  | Done |
| ⚠️     | Grading-critical / mandatory |
| 🏆     | Bonus / extra credit |

---

## Phase 0 — Project Setup & Reproducibility ⚠️

- [x] Create GitHub repo + clean structure
- [x] `.gitignore` (geodata + large rasters excluded)
- [x] `requirements.txt` exists
- [ ] Reproducible install test on a clean env (fresh venv) ⚠️
- [ ] Add a single “run-all” entrypoint (Makefile / run.sh) 🏆
- [ ] Ensure no raw data is in git history ⚠️

---

## Phase 1 — Problem Framing & Scope Definition ⚠️

- [x] Define land-cover classes (WorldCover → 6 final classes)
- [x] Define spatial unit: **100m × 100m**
- [x] Define temporal setup:
  - Labels: 2020 (v100), 2021 (v200)
  - Unlabeled: 2022–2025 (Sentinel-only, forward prediction for dashboard)
- [x] Define target task:
  - Multi-output regression: predict 6 proportions per cell (sum≈1)
  - Change Δ derived from predictions: Δ = p(T2) − p(T1)
- [x] Define intended user + non-allowed decisions (limitations)
- [ ] Write **Scope Document** (`reports/scope.md`) ⚠️
  - Must include the *algorithm version shift* caveat (v100 → v200)

---

## Phase 2 — Data Acquisition & Preprocessing ⚠️

### 2A — Sentinel-2 Seasonal Composites ⚠️
- [x] Download seasonal composites for **6 labeled periods**:
  - 2020: spring, summer, autumn
  - 2021: spring, summer, autumn
- [x] All 6 composites verified:
  - Geometry assertion: CRS/transform/shape match canonical anchor
  - Sizes ~85–88 MB each
- [x] Known runtime issues fixed in pipeline:
  - `rescale=False` stackstac dtype/scale conflict
  - dtype / NaN fill-value compatibility
  - dask-aware finite checks

### 2B — WorldCover Labels ⚠️
- [x] Download WorldCover 2020 v100 tile N48E009
- [x] Download WorldCover 2021 v200 tile N48E009
- [x] Reprojected to Sentinel anchor grid (nearest)
- [x] Class mapping (11 → 6 classes)
- [x] **Important limitation** ⚠️:
  - v100 vs v200 algorithm difference may create “fake change” signals

### 2C — InputQuality layer (optional)
- [x] Investigated InputQuality availability
- [x] **Decision: skip for now** (not public for our tile)
  - Use `valid_fraction` as the quality proxy
  - Can add InputQuality later without refactoring

### 2D — OSM Auxiliary Data 🏆
- [x] Downloaded OSM datasets (buildings/roads/landuse/natural/water)
- [x] Saved as GeoPackages in `data/raw/osm/`
- [ ] Decide final usage:
  - Option A: merge OSM once at final dataset stage (recommended)
  - Option B: suffix OSM per composite (creates lots of zero deltas)

---

## Phase 3 — Canonical Grid + Labels (Anchor-based) ⚠️

- [x] Create canonical anchor raster (Phase 1 artifact)
- [x] Build grid from anchor blocks:
  - **100m cells = 10×10 pixels**
  - **186 cols × 161 rows = 29,946 cells**
  - `cell_id` row-major, contiguous 0..N-1
  - Output: `data/processed/v2/grid.gpkg`
- [x] Aggregate label proportions per cell:
  - `data/processed/v2/labels_2020.parquet`
  - `data/processed/v2/labels_2021.parquet`
- [x] Compute change labels:
  - `data/processed/v2/labels_change.parquet` (delta = 2021−2020)

---

## Phase 4 — Feature Engineering (per composite) ⚠️

### 4A — Core features (done) ⚠️
- [x] Implement anchor-correct, join-safe, scale-safe extraction
  - Never drops `cell_id`
  - `cell_id → (row_idx, col_idx)` computed deterministically
  - Reflectance scale auto-detected (likely /10000) for texture assumptions
  - Control flags:
    - `valid_fraction`, `low_valid_fraction`
    - `reflectance_scale`
    - `full_features_computed`
- [x] Core feature set per composite:
  - Band stats: mean/std/min/max/median/q25/q75/finite_frac × 10 bands
  - Indices: NDVI/NDWI/NDBI/NDMI/NBR/SAVI/BSI/NDRE1/NDRE2 (stats)
  - Tasseled cap (brightness/greenness/wetness)
  - Spatial simple (edges, Laplacian, Moran’s I, NDVI spread)
- [x] Outputs produced for all 6 composites:
  - 29,946 rows each
  - **143 non-cell columns** (139 real features + 4 control cols)
  - Low valid fraction rates: ~0.2%–1.7% depending on season

### 4B — Full feature set (done) ⚠️
- [x] Full extraction completed (heavy features):
  - GLCM texture
  - Gabor
  - LBP histogram + entropy
  - HOG
  - Morphological profile features
  - Semivariogram features (+ fit params)
- [x] Full outputs verified:
  - All 6 composites: 29,946 cells × 239 columns (238 features + cell_id)
  - Identical `cell_id` set across all composites
  - `full_features_computed` = 1 for non-low-VF cells
  - Scale detected as 10000 across all composites

### 4C — Known fixed bugs (documentable as “data issues”) ⚠️
- [x] Bug 1: reflectance scaling mismatch (0..10000 vs 0..1 assumptions)
- [x] Bug 2: dropping cells based on quality threshold (join corruption risk)
- [x] Bug 3: cell_id order / mapping mismatch risk (row/col alignment)
- [x] Bug 4: imputation scope accidentally touching metadata columns

---

## Phase 5 — Merge + Delta Features (YOY + Seasonal Contrasts) ⚠️

### 5A — Core merge/deltas (done) ⚠️
- [x] Load all 6 composite feature tables
- [x] Build wide table with suffixes: `{feature}_{year}_{season}`
- [x] Compute deltas:
  - YoY: 2021−2020 per season → `delta_yoy_{season}_{feat}`
  - Seasonal contrasts within each year → `delta_{year}_{sB}_vs_{sA}_{feat}`
- [x] Output:
  - `data/processed/v2/features_merged_core.parquet`
  - **29,946 rows × 2,110 columns**
  - 0% NaN after imputation (expected; indicators preserve quality info)

### 5B — Full merge/deltas (done) ⚠️
- [x] Run compute_deltas.py on `feature-set=full`
- [x] Output:
  - `data/processed/v2/features_merged_full.parquet`
  - **29,946 rows × 3,535 columns** (660 MB)
  - 1,429 base + 702 YoY + 1,404 seasonal deltas
  - 0% NaN after imputation
- [x] Defensive checks:
  - Year ordering sorted (prevent sign flip)
  - Feature set intersection across composites (warn on mismatch)
  - Delta symmetry spot-checked on random cells

---

## Phase 6 — EDA + Reality Check ⚠️

> Must identify ≥3 non-trivial data issues. Must choose 1 you do NOT fix and justify why.

- [x] EDA script (`scripts/run_eda.py`) producing report-ready figures ⚠️
  - Feature distributions (hist/violin/box) — Fig 1, 4a, 4b
  - Correlation heatmaps (top-20 Spearman) — Fig 5a
  - Label distribution + label-change distribution — Fig 1, 2
  - Spatial maps of key features + Δlabels — Fig 3a, 3b
  - Redundancy clustering + drift (YoY + seasonal) — Fig 5b, 5c, 5d
  - Quality coupling analysis (all VF cols, all features) — Fig 6
  - Reflectance scale verification — Fig 7
  - Moran's I spatial autocorrelation (vectorized, full grid) — Fig 10
  - Engineering validation checks — Fig 9
  - Feature manifest (CSV + Parquet) — 12 tables total
- [x] Data issues (minimum 3) ⚠️
  - [x] Reflectance scale bug (fixed)
  - [x] Cell dropping / join corruption risk (fixed)
  - [x] Imputation scope (fixed)
  - [x] Choose one NOT fixed: **WorldCover v100→v200 label-version shift** ⚠️
    - Justify: cannot correct without alternative ground truth; treat as label noise + discuss
- [x] Save all figures to `reports/phase6/core/figures/` ⚠️
- [x] Hardened with 20+ defensive guardrails:
  - Row alignment + cross-table cell_id assertions
  - Label proportion invariants (sum-to-1, deltas sum-to-0)
  - CRS projection + grid geometry + row-major assumptions validated
  - NaN/Inf guards, numeric dtype filtering, empty-data guards
  - CLASS_COLORS/CLASS_NAMES/COMPOSITES consistency asserts

---

## Phase 7 — Train/Test Split Design (avoid spatial leakage) ⚠️

- [ ] Implement a spatially-aware split ⚠️
  - Recommended: blocked split by spatial tiles (e.g., 10×10 blocks or districts)
  - Store split indices to disk (reproducible)
- [ ] Baseline sanity: random split vs spatial split (show leakage effect)

---

## Phase 8 — Modeling ⚠️

> At least 2 models: one interpretable + one flexible.

- [ ] Decide target(s):
  - Option A: predict 2021 proportions from 2020 features (per season / merged)
  - Option B: predict Δ directly using merged features + deltas
- [ ] Model 1 (interpretable) ⚠️
  - Ridge/ElasticNet (multi-output) OR linear models per class
  - Coefficients + stability across folds
- [ ] Model 2 (flexible) ⚠️
  - RandomForest / XGBoost / LightGBM (multi-output via wrappers or per-class)
- [ ] Hyperparameter tuning (time-boxed) ⚠️
- [ ] Compare models (performance vs interpretability)

---

## Phase 9 — Evaluation Beyond Accuracy ⚠️

- [ ] Standard metrics (per class)
- [ ] Change-specific metrics ⚠️
  - false-change rate, stability in unchanged areas, Δ-magnitude calibration
- [ ] Stress tests (at least one) ⚠️
  - noise injection, missing features, seasonal shift, spatial shift
- [ ] Failure analysis maps (where model is wrong + why) ⚠️

---

## Phase 10 — Explainability + Uncertainty ⚠️

- [ ] Feature importance + SHAP (for flexible model)
- [ ] One helpful explanation + one misleading explanation ⚠️
- [ ] Uncertainty proxy:
  - ensemble variance / quantile models / bootstrap

---

## Phase 11 — Interactive Dashboard / Product ⚠️

- [ ] Streamlit + Folium (recommended)
- [ ] Time selection, class selection, Δ visualization
- [ ] Uncertainty overlay + disclaimer panel ⚠️
- [ ] Click cell → show features + prediction + explanation 🏆

---

## Phase 12 — Technical Report ⚠️ (≤ 10 pages)

- [ ] Report skeleton with required sections
- [ ] Insert EDA figures + tables
- [ ] Explicitly discuss:
  - spatial leakage
  - label version shift (v100 vs v200)
  - what’s not validated beyond 2021

---

## Phase 13 — ChatGPT Reflection ⚠️

- [ ] Log key prompts + outputs used
- [ ] “Arguing Against ChatGPT” case 1 + 2 (must be real, must have evidence)
- [ ] One misleading ChatGPT example (you already have good candidates from earlier bugs)

---

## Phase 14 — Demo Video ⚠️ (≤ 5 min)

- [ ] Script + record dashboard walkthrough
- [ ] Show: prediction, Δ, uncertainty, helpful vs misleading explanation, limitations

---

## Phase 15 — Repo Polish + Submission ⚠️

- [ ] End-to-end reproducible run (fresh environment) ⚠️
- [ ] Clean README: install, data fetch, pipeline steps, dashboard launch ⚠️
- [ ] Final deliverables check (report PDF, code, product, video, ChatGPT log) ⚠️
