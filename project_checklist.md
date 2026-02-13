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
  - **Grid size justification (retrospective analysis):**
    - Connected-component analysis of pixel-level change (2020→2021) in the AOI:
      76% of changed regions are ≤4 pixels, median = 2 px (200 m²)
    - At 100m (100 px/cell), these tiny artifacts → 1–4% proportion noise, naturally smoothed out
    - Real change (≥50 px regions, ~1.5% of regions) still produces 15–25% proportion shifts per cell
    - Smaller cells (e.g. 50m = 25 px) would amplify label artifacts to ~16% per cell
    - Larger cells (e.g. 200m) would halve sample count to ~7,500 — too few for spatial CV
    - **100m is the sweet spot: noise-smoothing + sufficient samples + spatial granularity**
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

- [x] Implement a spatially-aware split ⚠️
  - Tile-based blocked split: 10×10 cells (1 km²), 323 tile groups, 5-fold
  - Split indices saved to `data/processed/v2/split_spatial.parquet`
  - Metadata: `data/processed/v2/split_spatial_meta.json` (full repro params)
- [x] 6-way leakage comparison (Ridge regression) ⚠️
  - Random: R²=0.767 (baseline)
  - Grouped (scattered tiles): R²=0.742 (gap +0.025)
  - Contiguous (row bands): R²=0.691 (gap +0.076)
  - Morton Z-curve: R²=0.527 (gap +0.240)
  - Region growing: R²=0.429 (gap +0.338)
  - Region growing + buffer: R²=0.386 (gap +0.381)
- [x] Morton Z-curve fold builder (bit-interleaved, explicit uint64)
- [x] Multi-start region growing fold builder (10 restarts, balance+contiguity scoring)
- [x] Chebyshev buffer exclusion zone (configurable buffer_tiles)
- [x] Buffer sweep figure (R² vs separation distance)
- [x] Contiguity/balance/compactness metrics (`compute_fold_metrics`, BFS on tile graph)
  - Contiguous: connected=YES, max_dev=34.8%, compactness=1.878
  - Morton: connected=NO (fragments on 17×19), max_dev=27.9%, compactness=1.765
  - Region growing: connected=YES, max_dev=2.1%, compactness=1.871
- [x] Exported tables: `leakage_comparison.csv`, `buffer_sweep.csv`, `fold_contiguity.csv`
- [x] Fold maps: grouped, contiguous, Morton, region growing
- [x] Unit tests: 14 invariant tests in `tests/test_splitting.py` (all pass in ~3s)
  - Partition correctness, tile integrity, connectedness, balance, determinism,
    buffer no-overlap, n_starts overflow, metrics schema
- [x] Hardening fixes applied:
  - Morton: explicit uint64 + bit extraction (no signed overflow risk)
  - Region growing: multi-start + scoring, n_starts guard, tile integrity assert
  - Leakage comparison: fold-size guard for degenerate folds
  - Buffer_m: `max(tile_h, tile_w)` for non-square tile safety
  - Compactness: isoperimetric scaling `boundary_edges / sqrt(n_tiles)`
  - Metadata: region growing params recorded for reproducibility

## Phase 8 — Modeling ⚠️

> At least 2 models: one interpretable + one flexible.

- [x] Target: predict 2021 proportions from merged 2020+2021 seasonal features
- [x] Model 1 (interpretable) ⚠️ — Ridge baseline
  - Ridge CV R²≈0.43 (region growing + buffer) — anchors "no leakage" claim
- [/] Model 2 (flexible) ⚠️ — Trees + MLPs
- [/] Hyperparameter tuning (time-boxed) ⚠️
- [ ] Compare models (performance vs interpretability)

### Model comparison (all evaluated with spatial CV + buffer)

| Model | R² (uniform) | MAE (pp) | Notes |
|-------|-------------|----------|-------|
| Ridge | 0.527 | 5.21 | Interpretable baseline |
| ElasticNet | 0.434 | 5.43 | Sparse, but worse than Ridge |
| ExtraTrees | 0.675 | 3.34 | Best tree model (2109 feat) |
| RF | 0.684 | 3.23 | Competitive tree model |
| CatBoost | 0.671 | 4.00 | Gradient boosted, lag on MAE |
| **MLP (Phase 8 best)** | **0.752** | **2.92** | Plain MLP, single fold |

### Tree results (28 configs, complete — `trees_reduced_features.csv`)
- Best: **ExtraTrees R²=0.692**, MAE=3.13pp (`all_core`, 2109 features)
- RF: R²=0.684 | CatBoost: R²=0.671 (but best Aitchison=4.19)
- Feature sets: `all_core` barely beats `bands_and_indices` (798 feats)
- Trees plateau at ~0.69 regardless of ensemble size or feature set

### MLP V2 sweep (superseded by V4)
- Best: **relu L5 h256 R²=0.858**, MAE=2.55pp (`bands_indices`, 798 feats)
- MLP–Tree gap: **+0.17 R²** — MLPs learn cross-spectral interactions trees cannot
- ⚠️ V2 had BatchNorm confound on ReLU — fixed in V4

### MLP V4 search sweep (complete — 1549/1584 configs, fold-0 only)
- Script: `scripts/run_mlp_overnight_v4.py --stage search` (fold-0, 120 epochs)
- Best overall: **`residual_gelu_L6_d512_bn` R²=0.8634**, MAE=2.64pp
- Feature set ablation (best R² per set):
  | Rank | Feature Set | Features | Best R² |
  |------|---|---|---|
  | 1 | `bands_indices` | 798 | **0.8634** |
  | 2 | `bands_indices_glcm_lbp` | ~900 | 0.8598 |
  | 3 | `bands_indices_texture` | ~1100 | 0.8432 |
  | 4 | `full_no_deltas` | ~1400 | 0.8313 |
  | 5 | `bands_indices_hog` | ~850 | 0.8309 |
  | 6 | `all_full` | 3535 | 0.7952 |
  | 7 | `top500_full` | 500 | 0.6841 |
  | 8 | `texture_all` | ~300 | 0.6340 |
- Key finding: **more features = worse performance** — textures add noise at 100m resolution
- Architecture insights: residual + GELU + BatchNorm + lower LR (0.0005) wins
- Auto-CV launcher: `scripts/launch_cv_after_search.py` polls search → selects
  top 20 global + top 10 per feature set → launches CV (300 epochs × 5 folds)

### MLP V4 CV (complete — 83 configs × 5 folds = 415 runs)
- [x] CV complete
- Top 5 configs (5-fold mean ± std):
  | Rank | Config | R² mean | R² std | MAE | Epochs |
  |------|--------|---------|--------|-----|--------|
  | 1 | `bands_indices_glcm_lbp_plain_mish_L5_d512_bn` | **0.775** | 0.074 | 2.49 | 300 (cap!) |
  | 2 | `bands_indices_residual_silu_L10_d256_nonorm` | **0.772** | 0.050 | 2.52 | 297 (cap!) |
  | 3 | `full_no_deltas_plain_gelu_L5_d512_bn` | **0.771** | 0.035 | 2.65 | 280 |
  | 4 | `bands_indices_plain_silu_L5_d512_bn` | **0.771** | 0.054 | 2.45 | 300 (cap!) |
  | 5 | `bands_indices_texture_plain_silu_L5_d256_bn` | 0.767 | 0.033 | 2.62 | 295 |
- Key observations:
  - Many top configs hit the **300-epoch cap** — potentially undertrained
  - `bands_indices` (798 feat) and `bands_indices_glcm_lbp` (924 feat) dominate
  - Plain architectures (mish, silu) + BN competitive with residual deep networks
  - Fold-0 R²=0.86 drops to CV mean 0.77 → **~0.09 fold variance** indicates spatial heterogeneity
- [/] Final model selection from CV results

### MLP V5 deep training (in progress)
- Script: `scripts/run_mlp_v5_deep_train.py`
- Goal: let models train to convergence (2000 epoch cap, 5000-step patience)
- 60 configs × 5 folds = 300 runs (selected from V4 winners + new architectures)
- Tiers: proven winners, deep (12-16 blocks), GeGLU recovery, wide (d1024)
- [ ] V5 results pending

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
- [/] Uncertainty proxy:
  - [x] Conformal prediction intervals computed (6 models × 6 classes)
  - Ridge: 78–83% coverage, wide intervals (5–29pp)
  - MLP: 71–80% coverage, tighter intervals (0.03–14pp)
  - Tree models: 65–80% coverage, moderate intervals
  - ⚠️ Coverage below nominal 90% → conservative interpretation needed

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
