# 📋 Mapping Urban Change in Nuremberg — Full Project Checklist

> **Course**: Machine Learning WT 25/26 — Grabocka, Asano, Frey — UTN
> **Objective**: Build a tabular ML system that predicts land-cover composition and change in Nuremberg, deliver an interactive product with uncertainty communication.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]`  | Not started |
| `[/]`  | In progress |
| `[x]`  | Done |
| 👤     | Assign to a team member |
| ⚠️     | Grading-critical / mandatory |
| 🏆     | Bonus / extra credit |

---

## Phase 0 — Team Setup & Project Management

- [x] **Create a shared GitHub repository** (private until submission)
  - [ ] Add all team members as collaborators
  - [ ] Set up branch protection on `main` (require PR reviews)
  - [x] Add `.gitignore` (Python, Jupyter, geospatial data files, large rasters)
- [ ] **Set up project board** (GitHub Projects / Trello / Notion)
  - [ ] Create columns: Backlog → In Progress → Review → Done
  - [ ] Assign each Phase below as an epic with sub-tasks
- [x] **Agree on team conventions**
  - [x] Coding style (Black formatter, isort, flake8) → see `CONTRIBUTING.md`
  - [x] Commit message format (`ACTION: description`) → see `CONTRIBUTING.md`
  - [x] Notebook vs. script policy (scripts for pipeline, notebooks for exploration)
  - [x] Data storage policy (raw data **never** in Git; use `.gitignore` + download script)
- [x] **Create initial project structure** (all dirs created with `.gitkeep`)
- [x] **Set up a shared environment**
  - [x] Create `requirements.txt` with all dependencies
  - [x] All libraries installed (rasterio, geopandas, catboost, shap, streamlit, folium, etc.)
  - [ ] Verify all members can reproduce the environment

---

## Phase 1 — Problem Framing & Scope Definition ⚠️

> **Deliverable**: A 1–2 page "Scope Document" in `reports/scope.md` that the whole team agrees on.

- [x] **Define land-cover classes to focus on**
  - [x] Analyzed pixel distribution in Nuremberg tile (5,040,000 pixels)
  - [x] **6 final classes** (from 11 WorldCover classes):
    | Final Class | WorldCover Source | 2020 % | 2021 % | Justification |
    |------------|------------------|--------|--------|---------------|
    | Tree cover | Tree cover (10) | 46.6% | 47.8% | Dominant class, distinct spectral signature |
    | Grassland | Grassland (30) + Wetland (90) | 10.6% | 9.1% | Wetland (0.01%, 421 px) merged here — it's vegetated ground, not open water |
    | Cropland | Cropland (40) | 9.6% | 9.6% | Agricultural fringe of city, stable |
    | Built-up | Built-up (50) | 30.7% | 32.5% | Core urbanization signal |
    | Bare/sparse | Bare/sparse (60) | 1.7% | 0.2% | Kept separate — large drop signals construction→built-up transition |
    | Water | Water (80) | 0.8% | 0.8% | Pegnitz river, Wöhrder See — spectrally distinct |
  - [x] **Dropped**: Snow/ice, Mangroves, Moss/lichen, Shrubland — 0% presence in Nuremberg
- [x] **Define the spatial unit**: **100 m × 100 m grid cells**
  - [x] Data-driven analysis at 50/100/150/200/300/500m (see `notebooks/grid_size_analysis.py`)
  - [x] 100m chosen: 50,400 cells, 10×10 pixels per cell, 35% mixed-urban transition cells
  - [x] Justification: best balance of sample count (50K) vs. spatial resolution; 100 pixels per cell supports advanced feature extraction (Gabor wavelets, texture filters) beyond simple mean/std
- [x] **Define the temporal setup**
  - [x] Labeled periods: **2020** and **2021** (matching WorldCover labels)
  - [x] Unlabeled periods: **2022–2025** (Sentinel-2 only → forward prediction for dashboard)
  - [x] Prediction task: **Multi-output regression** — predict 6 class proportions per cell (sum to 1)
  - [x] Change derived by subtracting predicted proportions: Δ = proportions(T₂) − proportions(T₁)
- [x] **Define the intended user**
  - [x] City government / urban planning office — monitor built-up expansion, green space loss
  - [x] Infrastructure agencies — identify areas of rapid change for resource planning
  - [x] Environmental / ecology contractors — track vegetation loss, green corridor integrity
  - [x] Historians / researchers — document urban evolution over time
  - [x] Building / construction companies — identify development hotspots and trends
- [x] **Define what decisions must NOT be made based on your results** ⚠️
  - [x] Not suitable for parcel-level or property-level zoning decisions (100m resolution)
  - [x] Cannot detect individual buildings or distinguish building types
  - [x] Cannot assess legality of construction or land-use compliance
  - [x] Label differences between 2020/2021 may partly reflect algorithm changes (v100→v200), not real change
  - [x] Predictions beyond 2021 are extrapolations without ground-truth validation
- [ ] **Write the scope document** and get team consensus

---

## Phase 2 — Data Acquisition & Preprocessing

### 2A — Satellite Imagery ⚠️

- [x] **Acquire Sentinel-2 imagery for Nuremberg** — via Microsoft Planetary Computer (no auth)
  - [x] Nuremberg bbox: `[10.95, 49.38, 11.20, 49.52]`
  - [x] Downloaded 2020–2025 summer composites (Jun–Aug, <20% cloud, median composite)
  - [x] 10 bands: B02-B04, B05-B08, B8A, B11-B12 at 10 m, EPSG:32632
  - [x] Saved as GeoTIFF to `data/raw/`
  - [x] Download script: `scripts/download_all_data.py`

### 2B — ESA WorldCover Labels ⚠️

- [x] **Download ESA WorldCover 2020 (v100)** — tile N48E009 from AWS S3
- [x] **Download ESA WorldCover 2021 (v200)** — tile N48E009 from AWS S3
  - [x] Both saved to `data/labels/` (99.2 MB + 90.5 MB)

> [!WARNING]
> The 2020 and 2021 WorldCover maps used **different algorithm versions** (v100 vs v200). Changes between them may partly reflect algorithmic differences, not just real land-cover change. This **must** be discussed in your report.

- [ ] **(Fallback) Download CORINE Land Cover** 🏆
  - [ ] Only if WorldCover is unavailable for a needed year
  - [ ] Source: Copernicus Land Monitoring Service
  - [ ] Justify why CORINE was needed

### 2C — Optional Auxiliary Data 🏆

- [x] **OpenStreetMap features** — downloaded via `osmnx`
  - [x] 105,550 building footprints → `data/raw/osm/buildings.gpkg`
  - [x] 18,996 road segments + 8,083 intersections → `data/raw/osm/roads.gpkg`
  - [x] 3,380 land-use zones → `data/raw/osm/landuse.gpkg`
  - [x] 1,233 natural features + 241 water bodies → `data/raw/osm/natural.gpkg`, `water.gpkg`
- [ ] **Population / housing statistics**
  - [ ] Check availability from Bayerisches Landesamt für Statistik or Eurostat
  - [ ] Ensure spatial alignability with your grid
- [ ] **Environmental indicators** (e.g., NDVI time series, DEM/slope)

### 2D — Spatial Alignment & Preprocessing ⚠️

- [x] **Reproject all datasets to EPSG:32632 (UTM 32N)**
  - [x] Sentinel-2: already in EPSG:32632
  - [x] WorldCover: reprojected from EPSG:4326 using nearest-neighbor resampling
- [x] **Clip all rasters to Nuremberg boundary** (aligned to Sentinel-2 bounds)
- [x] **Create the spatial grid**: 100m x 100m, 186 cols x 161 rows = **29,946 cells**
  - [x] Saved as `data/processed/grid.gpkg`
  - [x] Each cell has exactly 100 valid pixels
- [x] **Aggregate labels per grid cell** — class proportions for 2020 and 2021
  - [x] `data/processed/labels_2020.parquet` (262 KB)
  - [x] `data/processed/labels_2021.parquet` (249 KB)
- [x] **Compute change labels** (delta = 2021 - 2020)
  - [x] `data/processed/labels_change.parquet` (283 KB)
  - [x] Key findings: built-up +1.8%, tree cover +1.2%, grassland -1.6%, bare/sparse -1.6%
- [x] **Pipeline script**: `src/data/build_grid.py`

---

## Phase 3 — Feature Engineering ⚠️

> All models must operate on **tabular / fixed-length feature vectors**. No raw images as input.

- [x] **Spectral features per grid cell** -- 50 features
  - [x] Per-band statistics: mean, std, min, max, median x 10 bands
- [x] **Spectral indices per grid cell** -- 25 features
  - [x] NDVI, NDBI, NDWI, SAVI, BSI (mean, std, median, q25, q75 each)
- [x] **Tasseled Cap transformation** -- 6 features
  - [x] Brightness, greenness, wetness (mean, std each)
- [x] **GLCM texture features** -- 10 features
  - [x] Contrast, homogeneity, energy, correlation, dissimilarity on NIR + NDVI
- [x] **Gabor wavelet features** -- 24 features
  - [x] 3 scales x 4 orientations x 2 stats (mean, std) on NIR band
- [x] **Spatial autocorrelation / edge features** -- 8 features
  - [x] Sobel edge density, Laplacian, Moran's I, NDVI spatial range/IQR
- [x] **OSM-derived features** -- ~40 features
  - [x] Building count/area, road length/count, water distance, land-use (one-hot)
- [x] **Compiled final feature matrix**: 29,440 cells x 163 features
  - [x] `data/processed/features_2020.parquet`
  - [x] `data/processed/features_2021.parquet`
  - [x] Pipeline script: `src/features/extract_features.py`

---

## Phase 4 — Data Exploration & Reality Check ⚠️

> **Must identify at least 3 non-trivial data issues**. Must explicitly choose one you do NOT fix and justify why.

- [ ] **Exploratory Data Analysis (EDA)** 👤
  - [ ] Distribution of each feature (histograms, box plots)
  - [ ] Correlation matrix / heatmap
  - [ ] Class distribution of land-cover labels (bar chart of proportions)
  - [ ] Spatial distribution maps of key features and labels
- [ ] **Identify data issues** (at least 3 required) ⚠️ 👤
  - [ ] **Issue 1 — Seasonal effects**: Imagery from different seasons → spectral differences not from actual change
    - [ ] Mitigation: Use summer composites from same month range for both years
  - [ ] **Issue 2 — Cloud cover & missing data**: Some cells may have no cloud-free observations
    - [ ] Mitigation: Use median composite, flag cells with few observations
  - [ ] **Issue 3 — Label noise in WorldCover maps**: Misclassification in ground-truth labels, algorithm version difference (v100 vs v200)
    - [ ] Mitigation: Discuss impact, possibly compare with CORINE for validation
  - [ ] **Issue 4 — Spatial resolution mismatch**: Feature resolution vs. label resolution misalignment
  - [ ] **Issue 5 — Spatial autocorrelation**: Nearby cells are correlated → standard cross-validation overestimates performance
- [ ] **Choose one issue you do NOT fix** ⚠️
  - [ ] State which issue and justify
  - [ ] Example: "We do not correct for spatial autocorrelation in our training split because ... however we acknowledge this may inflate our reported metrics."
- [ ] **Visualize change labels**
  - [ ] Map of Δ built-up, Δ vegetation across Nuremberg (not for modeling, just EDA)
  - [ ] Histogram of change magnitudes
- [ ] **Save all EDA figures** to `reports/figures/`

---

## Phase 5 — Modeling & Change Prediction ⚠️

> Min 2 models: one interpretable (required) + one flexible. No CNNs/Transformers.

### 5A — Train/Test Split Design

- [ ] **Design a spatial or temporal hold-out strategy** ⚠️ 👤
  - [ ] Option A: Spatial hold-out (e.g., hold out entire city districts / spatial blocks)
  - [ ] Option B: Temporal hold-out (if >2 time steps)
  - [ ] Option C: Spatial k-fold cross-validation (groups of spatially contiguous cells)
  - [ ] Justify the chosen strategy in the report
- [ ] **Implement the split**
  - [ ] Save train/validation/test indices
  - [ ] Ensure no spatial leakage between splits

### 5B — Model 1: Interpretable Model ⚠️

- [ ] **Choose model** 👤
  - [ ] Recommended: Ridge/Lasso Regression (for proportion prediction) or Logistic Regression (for change classification)
  - [ ] Alternative: Decision Tree, Elastic Net
- [ ] **Train on tabular features**
  - [ ] Predict land-cover composition at T₂, or Δ land-cover between T₁ and T₂
- [ ] **Tune hyperparameters** (cross-validation on training set)
- [ ] **Extract interpretability insights**
  - [ ] Feature importance / coefficients
  - [ ] Partial dependence plots or similar
- [ ] **Justify model choice** in report

### 5C — Model 2: Flexible Nonlinear Model ⚠️

- [ ] **Choose model** 👤
  - [ ] Recommended: Random Forest, XGBoost, LightGBM
  - [ ] Alternative: MLP (scikit-learn `MLPRegressor` / `MLPClassifier`)
- [ ] **Train on the same features**
- [ ] **Tune hyperparameters** (cross-validation)
- [ ] **Compare with Model 1**
  - [ ] Is the improvement worth the loss of interpretability?
- [ ] **Justify model choice** in report

### 5D — (Optional) Additional Models 🏆

- [ ] Simple temporal model over tabular features (e.g., autoregressive features)
- [ ] Ensemble of interpretable + flexible model
- [ ] Multi-output regression (predict all class proportions simultaneously)

### 5E — Justification Section ⚠️

- [ ] **Feature choices**: Why these features? What was excluded and why?
- [ ] **Model choices**: Why these models? What alternatives were considered?
- [ ] **Spatial and temporal resolution**: Why this grid size? Why these time steps?

---

## Phase 6 — Evaluation Beyond Accuracy ⚠️

> Must go beyond standard accuracy. Must include stress tests.

- [ ] **Standard metrics** 👤
  - [ ] If regression: MAE, RMSE, R² per land-cover class
  - [ ] If classification: Precision, Recall, F1, Confusion matrix
- [ ] **Spatial or temporal hold-out results** ⚠️
  - [ ] Report performance on the held-out spatial/temporal split
- [ ] **Change-specific metric** ⚠️ 👤
  - [ ] False change rate: How often does the model predict change where none occurred?
  - [ ] Stability metric: Does the model produce consistent predictions for unchanged areas?
  - [ ] Change detection accuracy (if binary change labels used)
- [ ] **Stress test (at least one)** ⚠️ 👤
  - [ ] **Feature noise test**: Add Gaussian noise to features → how much does performance degrade?
  - [ ] **Missing data test**: Randomly drop features → measure robustness
  - [ ] **Temporal shift test**: Train on one season, test on another
  - [ ] **Spatial shift test**: Train on one part of city, test on another
- [ ] **Discussion: Where and why is the model likely wrong?** ⚠️
  - [ ] Identify geographic areas with poorest predictions (visualize on map)
  - [ ] Discuss likely causes (e.g., construction sites, shadowed areas, mixed-use zones)
- [ ] **Save all evaluation figures and tables** to `reports/figures/`

---

## Phase 7 — Explainability & Trust ⚠️

> Explain to a non-expert: what changed, where, and how confident the system is.

- [ ] **What changed & where** 👤
  - [ ] Generate a map showing predicted change (e.g., Δ built-up) with magnitude and direction
  - [ ] Highlight top-N areas of greatest change
- [ ] **Confidence / uncertainty communication** 👤
  - [ ] For regression: prediction intervals or ensemble disagreement
  - [ ] For classification: predicted probabilities + calibration analysis
  - [ ] For Random Forest / ensemble: use variance across trees as uncertainty
  - [ ] Visualize uncertainty on the map (e.g., opacity or hatching)
- [ ] **One helpful explanation** ⚠️
  - [ ] Example: "The model confidently predicts increased built-up area in district X, supported by high NDBI increase and new building footprints in OSM."
  - [ ] Why it's helpful: connects features to human-understandable causes
- [ ] **One potentially misleading explanation** ⚠️
  - [ ] Example: "SHAP says NDVI decreased → model predicts urbanization, but the decrease was due to drought, not construction."
  - [ ] Why it's misleading: feature attribution conflates correlation with causation
- [ ] **Feature importance / SHAP analysis**
  - [ ] Use SHAP (`shap` library) for the flexible model
  - [ ] Display feature importances for interpretable model

---

## Phase 8 — Mandatory ChatGPT Reflection ⚠️

> Two labeled sections required in the report.

- [ ] **Document ChatGPT usage throughout the project** 👤
  - [ ] Keep a log of key prompts and responses
  - [ ] Note where ChatGPT was used (code generation, EDA ideas, model selection, etc.)
  - [ ] Highlight one example where ChatGPT was misleading
- [ ] **"Arguing Against ChatGPT — Case 1"** ⚠️ 👤
  - [ ] Describe the situation (modeling / evaluation / interpretation decision)
  - [ ] What ChatGPT suggested
  - [ ] Why you disagreed and what you did instead
  - [ ] Evidence supporting your decision
- [ ] **"Arguing Against ChatGPT — Case 2"** ⚠️ 👤
  - [ ] Different scenario from Case 1
  - [ ] Same structure as above
- [ ] **Save ChatGPT usage log** as `reports/chatgpt_log.md`

---

## Phase 9 — Interactive Dashboard / Product ⚠️

> Must deliver a working, interactive system. Minimum features listed below.

- [ ] **Choose framework** 👤
  - [ ] Recommended: **Streamlit** + **Folium** (easiest for geospatial)
  - [ ] Alternatives: Gradio, interactive Jupyter notebook, or custom dashboard
- [ ] **Implement core features** ⚠️
  - [ ] **Map of Nuremberg**: Interactive map (Folium or Plotly) showing the spatial grid
  - [ ] **Time selection**: Dropdown / slider to select time period (T₁, T₂)
  - [ ] **Predicted land-cover visualization**: Color-coded grid cells by predicted land-cover composition
  - [ ] **Change visualization**: Show predicted Δ values with color ramp (e.g., red = more built-up, green = more vegetation)
  - [ ] **Uncertainty / limitation display**: Show confidence/uncertainty per cell (tooltip or overlay), include a visible disclaimer panel
- [ ] **Additional features** 🏆
  - [ ] Click on a cell → show detailed breakdown (feature values, predicted probabilities, SHAP explanation)
  - [ ] Side-by-side comparison of T₁ vs. T₂
  - [ ] Layer toggle for different land-cover classes
  - [ ] Download predictions as CSV/GeoJSON
- [ ] **Test the dashboard end-to-end**
  - [ ] Ensure it loads without errors
  - [ ] Test with different browsers
  - [ ] Confirm all visualizations render correctly
- [ ] **Write dashboard README** 👤
  - [ ] How to install and run: `pip install -r requirements.txt && streamlit run app.py`
  - [ ] Screenshots in `assets/`

---

## Phase 10 — Technical Report ⚠️

> Max 10 pages. Must cover all required components.

- [ ] **Structure the report** 👤
  1. **Introduction & Problem Framing** (≈1 page)
     - Context, land-cover classes, spatial unit, temporal setup, intended user, decision limitations
  2. **Data Description** (≈1.5 pages)
     - Sources (with citations), preprocessing, spatial alignment
  3. **Data Exploration & Reality Check** (≈1.5 pages)
     - EDA findings, 3+ data issues, the one issue NOT fixed + justification
  4. **Feature Engineering** (≈1 page)
     - Feature descriptions, feature dictionary reference
  5. **Modeling** (≈2 pages)
     - At least 2 models, training procedure, hyperparameters, justifications
  6. **Evaluation** (≈1.5 pages)
     - Hold-out results, change-specific metric, stress test, failure analysis
  7. **Explainability & Trust** (≈1 page)
     - Helpful explanation, misleading explanation, uncertainty communication
  8. **ChatGPT Reflection** (≈0.5 page)
     - Two "Arguing Against ChatGPT" cases
  9. **Conclusion & Limitations** (≈0.5 page)
- [ ] **Write each section** (assign sections to team members) 👤👤👤
- [ ] **Include all required figures** referenced from `reports/figures/`
- [ ] **Proofread and ensure ≤ 10 pages**
- [ ] **Export as PDF** to `reports/final_report.pdf`

---

## Phase 11 — Demo Video ⚠️

> 5-minute video demonstrating the running product.

- [ ] **Plan the video script** 👤
  - [ ] 0:00–0:30 — Introduction: problem, team, approach
  - [ ] 0:30–2:00 — Dashboard walkthrough: map, time selection, predictions, change visualization
  - [ ] 2:00–3:30 — Explain key findings: where did change happen? How confident is the model?
  - [ ] 3:30–4:30 — Show a helpful vs. misleading explanation
  - [ ] 4:30–5:00 — Limitations, caveats, and conclusion
- [ ] **Record the demo** 👤
  - [ ] Screen recording of the dashboard in action (OBS Studio / Loom / QuickTime)
  - [ ] Voiceover or on-screen annotations
- [ ] **Edit and finalize**
  - [ ] Ensure ≤ 5 minutes
  - [ ] Export as MP4 to `assets/demo_video.mp4`

---

## Phase 12 — Code Repository Polish ⚠️

> Must be reproducible and documented.

- [ ] **Clean up all code** 👤
  - [ ] Remove dead code, TODO comments, debug prints
  - [ ] Add docstrings to all functions and modules
  - [ ] Ensure consistent formatting (run Black + isort)
- [ ] **Ensure reproducibility** 👤
  - [ ] All random seeds set and documented
  - [ ] Pipeline can be run end-to-end from data download to dashboard
  - [ ] `requirements.txt` is complete and pinned
- [ ] **Write a comprehensive README.md** ⚠️ 👤
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] Data download instructions (with links)
  - [ ] How to run the pipeline
  - [ ] How to run the dashboard
  - [ ] Project structure overview
  - [ ] Team members and contributions
  - [ ] Citations and data sources
- [ ] **Add a `Makefile` or `run.sh`** (optional but polished) 🏆
  - [ ] `make data` — download and preprocess data
  - [ ] `make features` — run feature engineering
  - [ ] `make train` — train models
  - [ ] `make dashboard` — launch the dashboard
- [ ] **Final git housekeeping**
  - [ ] Squash/rebase messy commits on feature branches
  - [ ] Ensure no large data files in git history
  - [ ] Tag the final release (e.g., `v1.0`)

---

## Phase 13 — Final Review & Submission

- [ ] **Cross-check all deliverables** ⚠️
  - [ ] ✅ Technical report (PDF, ≤ 10 pages)
  - [ ] ✅ Code repository (reproducible, documented)
  - [ ] ✅ Running interactive product (Streamlit app or equivalent)
  - [ ] ✅ 5-minute demo video
  - [ ] ✅ ChatGPT usage log (prompts, disagreements, one misleading example)
- [ ] **Peer review within team**
  - [ ] Each member reviews another's sections
  - [ ] Run the full pipeline from scratch on a clean environment
- [ ] **Submit through the required channel**
  - [ ] Make the GitHub repo accessible to instructors (or submit a zip)
  - [ ] Upload report and video as specified
- [ ] **Prepare for potential Q&A / presentation**
  - [ ] Each member should be able to explain any part of the project

---

## Quick Reference: Mandatory Datasets

| Dataset | Source | Resolution | Years | Usage |
|---------|--------|-----------|-------|-------|
| Sentinel-2 L2A | Google Earth Engine / Copernicus Open Access Hub | 10–20 m | 2020, 2021+ | Feature extraction (spectral bands) |
| ESA WorldCover | AWS S3 / Zenodo / WorldCover Viewer | 10 m | 2020 (v100), 2021 (v200) | Ground-truth labels |
| CORINE Land Cover | Copernicus Land Monitoring Service | 100 m | 2018 | Fallback labels (if needed) |

## Quick Reference: Allowed Models

| Model Type | Example | Use Case |
|-----------|---------|----------|
| Linear Regression | Ridge, Lasso, Elastic Net | Interpretable baseline for proportions |
| Logistic Regression | sklearn LogisticRegression | Change classification |
| Tree Ensembles | Random Forest, XGBoost, LightGBM | Flexible nonlinear model |
| MLP | sklearn MLPRegressor/Classifier | Simple neural model on tabular data |
| ❌ NOT ALLOWED | CNNs, Transformers, end-to-end DL | — |

## Quick Reference: Key Python Libraries

| Library | Purpose |
|---------|---------|
| `earthengine-api` | Access Sentinel-2 via Google Earth Engine |
| `rasterio` | Read/write/manipulate raster GeoTIFFs |
| `geopandas` | Geospatial vector data manipulation |
| `shapely` | Geometric operations |
| `pyproj` | CRS transformations |
| `rasterstats` | Zonal statistics (aggregate raster values per polygon) |
| `osmnx` | Download OpenStreetMap features |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` / `lightgbm` | Gradient boosting models |
| `shap` | Model explainability |
| `folium` | Interactive maps |
| `streamlit` | Interactive dashboard framework |
| `streamlit-folium` | Embed Folium maps in Streamlit |
| `matplotlib` / `seaborn` | Static visualizations |

---

> [!TIP]
> **Team Workflow Suggestion**: Assign phases to pairs of team members. One person implements, the other reviews. Rotate pairs across phases to ensure everyone understands the full system.
