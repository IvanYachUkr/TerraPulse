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
- [x] Model 1 (interpretable) ⚠️ — Ridge regression
  - RidgeCV R²=0.423 ± 0.197 (region growing + buffer, bi_LBP features, 864f)
  - Performs poorly as expected — linear model cannot capture the complex nonlinear
    interactions between spectral bands, indices, and texture features that dominate
    land-cover prediction
  - Saved: `models/final_ridge/` (weights, scalers, OOF predictions, coefficients)
- [x] Model 2 (flexible) ⚠️ — MLP (champion) + LightGBM (best tree)
- [x] Hyperparameter tuning (time-boxed) ⚠️
  - 798 MLP runs across 14 experiment versions (V5–V17)
  - 480 LightGBM sweep runs (32 configs × 3 feature sets × 5 folds)
  - 28 sklearn tree configs (ExtraTrees, RF, CatBoost × 4 feature sets)
  - 565 HGBR-based feature ablation runs (5 ablation studies)
- [x] Compare models (performance vs interpretability)

### Final model comparison (all on spatial CV with region growing + 1-tile buffer)

| Model | R² (5-fold) | MAE (pp) | Features | Role |
|-------|------------|----------|----------|------|
| **MLP** (champion) | **0.787 ± 0.038** | **2.50** | 864 (bi_LBP) | Flexible — final model |
| **LightGBM** (best tree) | **0.736 ± 0.044** | **2.99** | 438 (VegIdx+RedEdge+TC+NDTI+IRECI+CRI1) | Flexible — tree baseline |
| ExtraTrees | 0.692 | 3.13 | 2109 (all_core), fold-0 only | Explored, superseded by LightGBM |
| RF | 0.684 | 3.23 | 2109 (all_core), fold-0 only | Explored, superseded by LightGBM |
| CatBoost | 0.671 | 4.00 | 2109 (all_core), fold-0 only | Too slow, superseded by LightGBM |
| **Ridge** | **0.423 ± 0.197** | **5.63** | 864 (bi_LBP) | Interpretable baseline |

Saved models: `models/final_mlp/` (5 fold weights + scalers + OOF), `models/final_tree/` (5 fold LightGBM + OOF), `models/final_ridge/` (5 fold models + coefficients + OOF)

---

### Feature analysis with HistGradientBoostingRegressor (HGBR)

Extensive feature ablation was conducted using sklearn's `HistGradientBoostingRegressor`
as a fast proxy model to understand the impact of individual feature groups and their
combinations. This analysis informed feature set selection for both trees and MLPs.

**Ablation studies conducted** (565 total runs across 5 scripts):
- `ablation_spectral_all_folds.py` — 95 runs: spectral bands vs indices vs combined
- `ablation_texture_all_folds.py` — 25 runs: GLCM, LBP, Gabor, HOG, morphological
- `ablation_new_indices.py` — 315 runs: novel spectral indices (NDTI, IRECI, CRI1, EVI2, GNDVI, MNDWI)
- `ablation_fusion_novel.py` — 80 runs: fusion of base groups with novel indices
- `ablation_best5_mixes.py` — 50 runs: optimal combinations of top-5 feature groups

**Key finding**: RedEdge + VegIdx + TC (348 features) achieves 98.6% of full bands+indices
performance with half the features. Adding novel indices (NDTI, IRECI, CRI1) brought the
best HGBR performance to the 438-feature set ultimately used for LightGBM.

**HGBR vs LightGBM**: sklearn's HistGradientBoostingRegressor performed substantially worse
than LightGBM on the same feature sets — this is why scikit-learn trees were not used for
the final tree model. LightGBM's more sophisticated leaf-wise growth and native categorical
support gave it a clear edge.

---

### Tree modeling: sklearn trees → LightGBM

**Sklearn trees (28 configs, fold-0 only → `trees_reduced_features.csv`)**
- Tested: ExtraTrees (500/1000 estimators), RF (500), CatBoost (1000/deeper)
- Feature sets: `all_core` (2109f), `bands_and_indices` (798f), `bands_only` (480f), `indices_only` (318f)
- Trees plateau at R²≈0.67–0.69 regardless of ensemble size, feature set, or model type
- `all_core` (2109f) barely beats `bands_and_indices` (798f): 0.692 vs 0.686
- Adding more features does not help — trees cannot exploit texture information
  that MLPs use effectively; texture features that boost MLP performance often
  *degrade* tree performance by adding noise to the split search
- **CatBoost was unacceptably slow**: 442s (1000 trees) to 2328s (deeper config)
  vs ExtraTrees 21–239s for similar or better R². CatBoost was dropped.

**V8 trees on glcm_lbp features (7 configs × 5 folds → `mlp_v8_trees_summary.csv`)**
- Best: ExtraTrees R²=0.660, RF R²=0.643, CatBoost R²=0.642
- Confirmed: texture features (GLCM+LBP) hurt tree performance vs spectral-only

**LightGBM sweep (32 configs × 3 feature sets × 5 folds = 480 runs → `lgbm_sweep.csv`)**
- Feature sets: VegIdx+RedEdge+TC (348f), +NDTI+CRI1 (408f), +IRECI (438f)
- Hyperparameters swept: n_estimators (200–2000), max_depth (4–unlimited),
  learning_rate (0.01–0.2), num_leaves (15–255), min_child_samples (5–100),
  reg_lambda (0–10), subsample (0.6–1.0), colsample_bytree (0.5–1.0)
- **Best: `strong_wide` config (R²=0.749, MAE=2.94)** — n_estimators=1000,
  max_depth=6, lr=0.03, num_leaves=255, reg_lambda=3.0, subsample=0.8, colsample=0.7
- Adding IRECI+CRI1 spectral indices improved R² from 0.717 (base, 348f) to 0.749 (438f)
- LightGBM achieves very high results for a tree model (R²=0.749), closing the
  gap with MLPs significantly compared to sklearn trees (0.69)

---

### MLP architecture and feature set exploration

**Total: 798 runs across 14 experiment versions (V5–V17)**

**Architectures tested:**
- **Types**: Plain (feedforward) vs Residual (skip connections)
  - Plain: 183 runs, best R²=0.864 — winner overall
  - Residual: 45 runs, best R²=0.834
- **Activations**: SiLU, Mish, GELU, GeGLU, ReLU
  - SiLU: best overall (R²=0.864), most stable
  - Mish: close second, slightly worse stability
  - GELU: competitive in residual networks
  - GeGLU: catastrophically unstable, dropped
- **Depths**: L3, L5, L7, L10, L12, L16, L20
  - Shallow L5 dominates; deep L12-L16 competitive but not better
  - Very deep L20 shows no benefit, just slower training
- **Hidden dims**: d256, d512, d1024, d1536, d2048
  - d1024 is the sweet spot (best R²=0.864)
  - d256 too small, d2048 adds no benefit with more compute
- **Normalization**: BatchNorm vs no-norm vs LayerNorm
  - BatchNorm always wins
- **Heads**: ILR+softmax (compositional) vs Dirichlet
  - ILR+softmax is the standard, Dirichlet explored but not better

**Feature sets tested for MLP:**

| Feature Set | # Features | Best R² (5-fold) | Notes |
|---|---|---|---|
| **bi_LBP** | 864 | **0.787** | **Champion** — bands+indices+LBP |
| bi_LBP_all5 | 1128 | 0.768 | + multi-band LBP, diminishing returns |
| bi_LBP_NIR | 864 | 0.768 | LBP on NIR only |
| bands_indices_glcm_lbp | 924 | 0.787 | ≈ bi_LBP (V5 naming) |
| bi_Gab2_DMP | 1566 | 0.758 | Complex Gabor + morph DMP |
| bi_LBP_mLBP | 1344 | 0.761 | + multi-band LBP (different subset) |
| bi_LBP_Gab2_DMP | 1632 | 0.745 | Kitchen sink — too many features |
| bands_indices | 798 | 0.769 | Spectral only, no texture |
| full_no_deltas | 1428 | 0.762 | All features except deltas |
| all_full | 3535 | 0.795 | Everything (fold-0 only) |
| bi_mLBP_only | 1278 | 0.623 | Multi-band LBP without base bands |

**Key MLP findings:**
- Texture features (LBP) **do help** MLPs: +0.018 R² (0.769 → 0.787)
- But adding too many texture types (Gabor, HOG, morph, semivariogram) hurts — noise
- Shallow-wide (L5 × d1024) with BatchNorm and SiLU is the winning architecture
- MLP–Tree gap: **+0.038 R²** (0.787 vs 0.749) over the best LightGBM — MLPs
  learn cross-spectral interactions that trees cannot model

---

### Training progression (MLP experiments V5–V17)

| Version | Focus | Configs | Runs | Key Finding |
|---|---|---|---|---|
| V4 search | Architecture sweep (fold-0) | 1549 | 1549 | residual+GELU+BN best on fold-0 |
| V4 CV | Top configs (5-fold) | 83 | 415 | 300-epoch cap → undertrained |
| V5 | Deep training (2000 ep) | 20 | 233 | glcm_lbp L5 d1024 BN = **0.787** |
| V5_arch | Architecture variants | 16 | 80 | Confirms plain > residual |
| V6 | More architectures | 10 | 50 | SiLU dominates |
| V7 | Architecture refinement | 10 | 50 | L5 optimal depth |
| V9 | Texture ablation | ~16 | 82 | LBP+GLCM best texture combo |
| V10 | Definitive sweep | 15 | 75 | bi_LBP confirmed champion |
| V11 | Gabor v2 + morph DMP | 7 | 42 | Complex textures no benefit |
| V12 | Multi-band LBP | 4 | 20 | Marginal improvement |
| V12b | Arch sweep on multi-LBP | 6 | 30 | d1024 optimal width |
| V13 | Clean multi-band LBP | 22 | 110 | Could not beat V10/V5 |
| V14 | V10 reproduce check | 1 | 5 | Confirmed (env issue found) |
| V15 | Per-patch LBP (Rust) | 2 | 10 | Boundary mode bug found |
| V16 | Rust reproduce | 1 | 5 | After Rust LBP fix |
| V17 | Multi-seed replication | 1×6 seeds | 6 | Seed 42 = best, R²=0.787 |

### Consolidated result tables
- [all_mlp_results.csv](reports/phase8/tables/all_mlp_results.csv) — 798 per-fold MLP runs
- [all_mlp_summary.csv](reports/phase8/tables/all_mlp_summary.csv) — 53 unique MLP configs
- [all_tree_results.csv](reports/phase8/tables/all_tree_results.csv) — 543 per-fold tree runs
- [all_tree_summary.csv](reports/phase8/tables/all_tree_summary.csv) — 125 unique tree configs

### Future: Rust-based feature extraction + inference 🏆
- [ ] Port LBP extraction to Rust (PyO3) for real-time inference
- [ ] ONNX model export for Rust-side inference (no Python at serving time)
- [ ] Docker deployment (multi-stage build → slim image)

---

## Phase 9 — Evaluation Beyond Accuracy ⚠️

- [x] Standard metrics (per class)
- [x] Change-specific metrics ⚠️
  - false-change rate, stability in unchanged areas, Δ-magnitude calibration
- [x] Stress tests (3 tests) ⚠️
  - Gaussian noise injection, season dropout, feature-group ablation
- [x] Failure analysis maps (where model is wrong + why) ⚠️

Script: `scripts/run_evaluation_phase9.py` — 14 figures + 7 CSV tables (65s runtime)

### A) Per-class metrics (MLP vs Ridge)

| Class | R² MLP | R² Ridge | MAE MLP (pp) | MAE Ridge (pp) |
|---|---|---|---|---|
| Tree Cover | 0.963 | 0.922 | — | — |
| Grassland | 0.844 | 0.691 | — | — |
| Cropland | 0.916 | 0.589 | — | — |
| Built-up | 0.961 | 0.881 | — | — |
| Bare/Sparse | 0.406 | 0.053 | — | — |
| Water | 0.910 | 0.769 | — | — |

- R² uniform: MLP=0.833, Ridge=0.651
- Aitchison: MLP=7.81, Ridge=11.20
- KL divergence: MLP=0.080, Ridge=0.275
- MLP dominates on every class; biggest gap on cropland (+0.33) and bare/sparse (+0.35)

### B) Change-specific metrics

- **False-change rate** (τ=5%): MLP=24%, Ridge=87%
- **Missed-change rate** (τ=5%): MLP=6.5%, Ridge=1.4%
- **Stability MAE** in unchanged cells: MLP=0.83pp, Ridge=4.19pp
- **Delta correlations** (Pearson): bare_sparse=0.96 (high), water=0.25 (worst)
- Ridge over-predicts change everywhere (87% false-change) because it can't model stable areas

### C) Stress tests

**Noise injection**: model robust to σ≤0.25 (R² drops 0.004), degrades at σ=0.5 (−0.02), collapses at σ=2.0 (R²=−0.06)

**Season dropout** (zeroing all features from one season):
| Season | R² | Delta |
|---|---|---|
| 2020_spring | −9.09 | −9.92 (most critical!) |
| 2021_summer | −1.81 | −2.64 |
| 2020_autumn | −0.65 | −1.49 |
| 2020_summer | −0.56 | −1.40 |
| 2021_spring | 0.00 | −0.83 |
| 2021_autumn | 0.35 | −0.48 (least critical) |

**Feature ablation** (zeroing feature groups):
- Drop Bands (480f): R²=−1.44 (−2.27) — catastrophic, bands are essential
- Drop LBP (66f): R²=0.47 (−0.36) — LBP provides +0.36 R² (biggest impact per-feature)
- Drop Indices (192f): R²=0.61 (−0.23) — indices complement bands

### D) Failure analysis

**Error by dominant class** (MLP MAE in pp):
- Tree Cover: 1.57pp (lowest — dominant class, most data)
- Built-up: 2.51pp (well predicted despite heterogeneity)
- Cropland: 4.66pp | Grassland: 4.89pp (confused with each other)
- Water: 5.12pp | Bare/Sparse: 13.81pp (rarest class, highest error)

**Fold variation**: R² ranges 0.753–0.844 across 5 folds; MAE 1.18–3.61pp

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

---

## Future: Rust Performance Rewrite 🏆

> For interactive "select any territory → predict" product. Not needed for course submission.

- [ ] Rust feature extraction module (via PyO3 or standalone binary)
  - Raster I/O: `gdal` crate (C-speed reads, no Python overhead)
  - Numerical: `ndarray` + BLAS (NumPy-equivalent)
  - Parallelism: `rayon` for data-parallel cell processing across all CPU cores
  - Gabor/Morph/LBP: port ~500 lines of Python to compiled Rust
  - Expected speedup: 20-50× over Python (CPU), 100×+ with GPU path
- [ ] ONNX model export + Rust inference via `ort` crate
  - Export trained PyTorch model → ONNX
  - Inference in Rust: no Python runtime needed at serving time
- [ ] Docker deployment
  - Multi-stage build: `rust:latest` → `debian:slim` (~30 MB final image)
  - Or fully static musl binary → `FROM scratch` (~10 MB image)
  - GDAL + PROJ packaged in build stage
- [ ] Alternative: PyTorch GPU batch extraction (simpler, stays in Python)
  - Process entire raster as one tensor, no per-cell loop
  - `F.conv2d` for Gabor, `F.max_pool2d` for morphology
  - Good enough for demo, ~2-5s per territory on GPU

