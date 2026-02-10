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

- [ ] **Create a shared GitHub repository** (private until submission)
  - [ ] Add all team members as collaborators
  - [ ] Set up branch protection on `main` (require PR reviews)
  - [ ] Add `.gitignore` (Python, Jupyter, geospatial data files, large rasters)
- [ ] **Set up project board** (GitHub Projects / Trello / Notion)
  - [ ] Create columns: Backlog → In Progress → Review → Done
  - [ ] Assign each Phase below as an epic with sub-tasks
- [ ] **Agree on team conventions**
  - [ ] Coding style (e.g., Black formatter, isort, flake8)
  - [ ] Commit message format (e.g., Conventional Commits)
  - [ ] Notebook vs. script policy (scripts for pipeline, notebooks for exploration)
  - [ ] Data storage policy (raw data **never** in Git; use `.gitignore` + cloud / shared drive)
- [ ] **Create initial project structure**
  ```
  project/
  ├── data/               # .gitignored — raw + processed data
  │   ├── raw/
  │   ├── processed/
  │   └── labels/
  ├── notebooks/          # exploratory analysis
  ├── src/                # production code
  │   ├── data/           # data download & preprocessing
  │   ├── features/       # feature engineering
  │   ├── models/         # model training & prediction
  │   ├── evaluation/     # metrics, stress tests
  │   └── dashboard/      # Streamlit / Gradio app
  ├── reports/            # technical report, figures
  ├── assets/             # images, demo video
  ├── tests/              # unit & integration tests
  ├── requirements.txt
  ├── README.md
  └── .gitignore
  ```
- [ ] **Set up a shared environment**
  - [ ] Create `requirements.txt` or `environment.yml` with pinned versions
  - [ ] Core libs: `earthengine-api`, `rasterio`, `geopandas`, `shapely`, `pyproj`, `folium`, `streamlit`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `matplotlib`, `seaborn`, `osmnx` (optional)
  - [ ] Verify all members can reproduce the environment

---

## Phase 1 — Problem Framing & Scope Definition ⚠️

> **Deliverable**: A 1–2 page "Scope Document" in `reports/scope.md` that the whole team agrees on.

- [ ] **Define land-cover classes to focus on** 👤
  - [ ] Decide which ESA WorldCover classes to group/merge (e.g., Built-up, Tree cover, Grassland, Cropland, Water, Other)
  - [ ] Justify why these classes matter for Nuremberg's urban context
- [ ] **Define the spatial unit** 👤
  - [ ] Choose grid cells (e.g., 100 m × 100 m, 250 m × 250 m, 500 m × 500 m) vs. hexagons vs. districts
  - [ ] Justify the choice considering data resolution (Sentinel-2 = 10 m, WorldCover = 10 m)
- [ ] **Define the temporal setup** 👤
  - [ ] Select at least **two time periods** (e.g., 2020 and 2021 using ESA WorldCover; extend with Landsat if going earlier)
  - [ ] Define what the prediction task is:
    - Predict land-cover composition at time T₂ from features at T₁?
    - Predict land-cover change (Δ) between T₁ and T₂?
    - Both?
- [ ] **Define the intended user** 👤
  - [ ] Who is the target audience? (e.g., city planners, environmental agencies, citizens)
  - [ ] What decisions would they make with this system?
- [ ] **Define what decisions must NOT be made based on your results** ⚠️
  - [ ] State limitations explicitly (e.g., not suitable for parcel-level zoning, cannot detect illegal building activity, etc.)
- [ ] **Write the scope document** and get team consensus

---

## Phase 2 — Data Acquisition & Preprocessing

### 2A — Satellite Imagery ⚠️

- [ ] **Acquire Sentinel-2 imagery for Nuremberg** 👤
  - [ ] Set up Google Earth Engine account and authenticate (`ee.Authenticate()`)
  - [ ] Define Nuremberg bounding box / administrative boundary polygon
    - Nuremberg approx. bbox: `[10.95, 49.38, 11.20, 49.52]` (lon/lat)
  - [ ] Download Sentinel-2 Level-2A (surface reflectance) for **Time Period 1** (e.g., summer 2020)
    - Filter by cloud cover < 10–20%
    - Create cloud-free composite (median composite)
    - Bands: B2 (Blue), B3 (Green), B4 (Red), B5-B7 (Red Edge), B8 (NIR), B8A (Narrow NIR), B11-B12 (SWIR)
  - [ ] Download Sentinel-2 for **Time Period 2** (e.g., summer 2021)
    - Same filters and process
  - [ ] Export as GeoTIFF to Google Drive, then download locally to `data/raw/`
  - [ ] Document exact GEE scripts / parameters used
- [ ] **(Optional) Acquire Landsat imagery for earlier years** 🏆 👤
  - [ ] If extending temporal coverage beyond 2020–2021
  - [ ] Be aware of resolution differences (30 m vs. 10 m) — discuss spatial alignment

### 2B — ESA WorldCover Labels ⚠️

- [ ] **Download ESA WorldCover 2020 (v100)** 👤
  - [ ] From AWS S3, Zenodo, or WorldCover Viewer
  - [ ] Identify the correct 3×3° tile covering Nuremberg
  - [ ] Save to `data/labels/`
- [ ] **Download ESA WorldCover 2021 (v200)** 👤
  - [ ] Same process
  - [ ] Save to `data/labels/`

> [!WARNING]
> The 2020 and 2021 WorldCover maps used **different algorithm versions** (v100 vs v200). Changes between them may partly reflect algorithmic differences, not just real land-cover change. This **must** be discussed in your report.

- [ ] **(Fallback) Download CORINE Land Cover** 🏆
  - [ ] Only if WorldCover is unavailable for a needed year
  - [ ] Source: Copernicus Land Monitoring Service
  - [ ] Justify why CORINE was needed

### 2C — Optional Auxiliary Data 🏆

- [ ] **OpenStreetMap features** 👤
  - [ ] Use `osmnx` to extract for Nuremberg:
    - [ ] Building footprints → compute building density per grid cell
    - [ ] Road network → compute road density / intersection density per grid cell
    - [ ] Land use polygons (parks, industrial areas, residential, etc.)
  - [ ] Save as GeoDataFrames / shapefiles
- [ ] **Population / housing statistics**
  - [ ] Check availability from Bayerisches Landesamt für Statistik or Eurostat
  - [ ] Ensure spatial alignability with your grid
- [ ] **Environmental indicators** (e.g., NDVI time series, DEM/slope)

### 2D — Spatial Alignment & Preprocessing ⚠️

- [ ] **Reproject all datasets to a common CRS** 👤
  - [ ] Recommended: EPSG:32632 (UTM zone 32N, covers Nuremberg)
  - [ ] Use `rasterio` or `geopandas` for reprojection
- [ ] **Clip all rasters to Nuremberg boundary**
  - [ ] Use a vector boundary (from OSM or official administrative boundaries)
- [ ] **Create the spatial grid** 👤
  - [ ] Generate grid cells (e.g., 200 m × 200 m) covering Nuremberg
  - [ ] Each grid cell = one sample in the tabular dataset
  - [ ] Assign unique IDs to each cell
- [ ] **Aggregate labels per grid cell**
  - [ ] For each cell, compute the **proportion of each land-cover class** from WorldCover at T₁ and T₂
  - [ ] Compute **change labels** (Δ built-up, Δ vegetation, etc.)
  - [ ] (Optional) Derive binary change/no-change labels with a justified threshold
- [ ] **Save processed datasets** to `data/processed/` as Parquet / GeoParquet / CSV

---

## Phase 3 — Feature Engineering ⚠️

> All models must operate on **tabular / fixed-length feature vectors**. No raw images as input.

- [ ] **Spectral features per grid cell** 👤
  - [ ] Per-band statistics: mean, median, std, min, max of reflectance within each cell
  - [ ] Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- [ ] **Spectral indices per grid cell** 👤
  - [ ] NDVI = (NIR − Red) / (NIR + Red) — vegetation vigor
  - [ ] NDWI = (Green − NIR) / (Green + NIR) — water bodies
  - [ ] NDBI = (SWIR − NIR) / (SWIR + NIR) — built-up areas
  - [ ] EVI, SAVI (optional but recommended)
  - [ ] Compute statistics (mean, std) per cell for each index
- [ ] **Texture features** (optional but valuable) 👤
  - [ ] GLCM-derived features (contrast, homogeneity, entropy) per band or NDVI
  - [ ] Standard deviation of spectral bands within each cell (a simple texture proxy)
- [ ] **Temporal features** 👤
  - [ ] Change in spectral features between T₁ and T₂ (Δ NDVI, Δ NDBI, etc.)
  - [ ] Ratio features (T₂ / T₁ spectral values)
- [ ] **Spatial context features** (optional) 🏆
  - [ ] Neighboring cell statistics (spatial lag features)
  - [ ] Distance to city center, major roads, water bodies
- [ ] **OSM-derived features** (optional) 🏆 👤
  - [ ] Building density, road density, land-use type per cell
- [ ] **Compile final feature matrix**
  - [ ] One row per grid cell (and per time step if predicting composition)
  - [ ] Save as `data/processed/features.parquet`
  - [ ] Document all features in a feature dictionary (`reports/feature_dictionary.md`)

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
