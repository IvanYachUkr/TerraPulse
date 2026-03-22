#import "ieee_layout.typ": ieee
#import "@preview/note-me:0.6.0": *

#show link: underline

#show: ieee.with(
  title: [🌍 TerraPulse - Final Project Report WS25/26],
  abstract: [
        We´re excited to present TerraPulse, our machine learning-based application for predicting land-cover composition and land cover change based on the data from ESA WorldCover 2020 and 2021. This project focuses on the city of Nuremberg as its primary case study, while also showcasing its performance around the globe.

    Explore the world, tile by tile, with our publicly available _TerraPulse_ app now: @containerapp

  ],
  authors: (
    (
      name: "Clemens Roßkopf",
      organization: [University of Technology Nuremberg],
      email: "clemens.rosskopf@utn.de",
    ),
    (
      name: "Ivan Iachnyk",
      organization: [University of Technology Nuremberg],
      email: "ivan.iachnyk@utn.de",
    ),
    (
      name: "Robin Sternberg",
      organization: [University of Technology Nuremberg],
      email: "robin.klemens.elias.sternberg@utn.de",
    ),
    (
      name: "Claudius Kühn",
      organization: [University of Technology Nuremberg],
      email: "claudius.kuehn@utn.de",
    ),
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Figure],
)



// Define code style
#let code(body) = [
  #block(
    fill: rgb("#f7f7f7"),
    inset: 6pt,
    width: 100%, // full column width
    radius: 4pt, // rounded corners
    stroke: 0.5pt + rgb("#ccc"), // optional border
  )[
    #body
  ]
]
= TODO LIST:
- General:
  - Look through Repo. Index EVERY SINGLE ONE of your written files/scripts/yamls and add which Model (1,2,3,4) it works for 

  
- Ivan:
  - Class distribution diagram Train, Eval, Test. Mention / put in Model 2 chapter

- Clemens:
  - Diagramms to visualize and make it easier to grasp
  - What kind of hyperparameter tuning did you use?

- Robin:
  - Make one short table summarizing what each model does and where it is used on the dashboard and put it between the models-chapter header and the "Model 1" part.


= Introduction

This report outlines _TerraPulse_, our machine learning-based application for predicting land-cover composition and land-cover change, with a particular focus on Nuremberg, while also enabling global predictions.

The ESA WorldCover datasets of the years 2020 @worldcover2020 and 2021 @worldcover2021 are used for training multiple models in order to predict:
+ land-cover classification labels
+ change likelihoods
+ new land-cover classification labels

As a result, we're able to derive the land-cover composition of a satellite image and provide the land-cover change when comparing label classification from multiple satellite images of different years. We even experimented with predicting future label changes within Nuremberg.

In general, sectors like urban planning, environmental monitoring, climate policy as well as business decisions rely on land-cover data.
As we've been instructed to focus on Nuremberg, the first tab of the dashboard shows the map of Nuremberg with its districts to provide easy access via the city´s structure level.
_TerraPulse_ is also useful for users outside Nuremberg, as the second tab, "Global," lets them select any region worldwide, run the classification pipeline across multiple years, and evaluate the resulting land-cover labels.


= Technical Stack

== Machine learning

All training and experimentation was done in *Python*.
The pixel-wise classification model (Model 1) uses *CatBoost* with GPU training (CUDA).
The global MLP (Model 2) is built in *PyTorch* (mixed-precision FP16, CUDA) and exported to *ONNX* for deployment.
Hyperparameter optimization uses *Optuna* (Model 1) and a custom *BOHB* sweep (Model 2).
#todo[Does Clemens need to add his HPO here as well?]
Geospatial data access relies on *pystac-client* and *stackstac* (Microsoft Planetary Computer STAC API), with *rasterio*, *geopandas*, and *xarray* for raster I/O and coordinate transforms.

At inference time, a standalone *Rust* binary (_terrapulse_) replaces the entire Python pipeline.
It downloads Sentinel-1/2 imagery, builds cloud-free composites, extracts all 1,764 features per cell, and runs the ONNX model.
Parallelism is provided by *Rayon* (CPU) and ONNX Runtime (the _ort_ crate with dynamic loading).

== Application

The dashboard is a single-page application.

*Frontend*: React 19 + TypeScript, built with Vite.
Maps are rendered with *deck.gl* (GPU-accelerated WebGL layers) on top of *MapLibre GL*.
Charts use *Chart.js* via react-chartjs-2.

*Backend*: A *FastAPI* server (Python, Uvicorn) exposes a REST API.
For the Global tab, the API spawns the Rust _terrapulse_ binary as a subprocess to run the full satellite-download → feature-extraction → inference pipeline on demand.
Precomputed Nuremberg data is served directly from Parquet/JSON files.

*Deployment*: A multi-stage *Docker* image (Rust build → Node.js build → Python runtime) packages everything into a single container, deployed on *Azure Container Apps*.


= Data

All data used in this project is publicly available and does not require any pre-authorized accounts or OAuth credentials, which was a deliberate choice so the inference pipeline can run fully autonomously on any machine.

== Data Sources

*Sentinel-2 Level-2A* surface reflectance imagery is the primary input for both the classification and change-prediction models.
We query it through the Microsoft Planetary Computer STAC API @planetary-computer, which provides free, anonymous access to the full Copernicus Sentinel archive.
Per scene, we download bands B02--B08, B8A, B11, B12 (10--20 m resolution) and the L2A Scene Classification Layer (SCL) used by our declouding pipeline (described here: @declouding-description.
Scenes are searched per season (spring, summer, autumn) with a cloud-cover ramp (40%→50%→60%) and a ±14-day date expansion fallback when a season has too few usable acquisitions.

*Sentinel-1 GRD (IW mode)* C-band SAR backscatter complements the optical data, particularly for situations where persistent cloud cover makes optical composites unreliable.
Like Sentinel-2, it is accessed via the Planetary Computer STAC API @planetary-computer with no additional credentials.

We prefer ascending orbit scenes for consistency, falling back to any orbit when fewer than three ascending scenes are available within the seasonal window.
This fallback can trigger for small or edge-of-swath bounding boxes, non-European regions with sparser coverage, or any query after December 2021 when the failure of Sentinel-1B halved the constellation's revisit frequency.


*ESA WorldCover 10 m* land-cover maps for 2020 (v100) @worldcover2020 and 2021 (v200) @worldcover2021 serve as our ground-truth labels.
The GeoTIFF tiles are downloaded directly from the public ESA WorldCover S3 bucket @esa-worldcover-s3.
We map the original 11 ESA classes to a reduced set of 7: tree cover, shrubland, grassland (merging with herbaceous wetland, which is spectrally near-identical and too rare in our training regions to learn as a separate class), cropland, built-up, bare/sparse vegetation, and water.


*LUCAS 2022 Survey* @lucas-2022 point observations were used for manual cross-checking of ESA WorldCover labels in ambiguous cases (see @rare-labels below).

*Nuremberg District Statistics* @nuremberg_district_statistics by the City of Nuremberg.

*Nuremberg District Shapefiles* @nuremberg_district_shapefiles by the City of Nuremberg. Used for the mouse hover feature on the Nuremberg Tab.

We did not limit the product to specific land-cover classes and kept the 10 m pixel grid as spatial unit, aggregating to higher resolutions with a zoom-slider (exception: Model 2 where we natively predict 10x10 pixel-supercells)


= Feature Engineering <feature-engineering>

All models share a common feature engineering pipeline built on Sentinel-2 L2A surface reflectance.
The base input is 10 spectral bands: B02 (blue), B03 (green), B04 (red), B05–B07 (red-edge 1–3), B08 (NIR), B8A (narrow NIR), B11 (SWIR1), B12 (SWIR2).
20m bands (B05, B06, B07, B8A, B11, B12) are upsampled to 10m to match the native pixel grid.
For the cell-level models, 20m bands within each 10×10 pixel cell are block-reduced to a 5×5 grid of super-pixels (each covering 20m×20m, matching the native sensor resolution) via 2×2 mean pooling before computing statistics.

== Spectral indices (deployed)

15 normalized indices are computed per pixel (or per cell).
Model 1 uses 9 of these; Model 2 uses all 15.

=== Vegetation indices

$ "NDVI" = (B_08 - B_04) / (B_08 + B_04) $
$ "EVI2" = 2.5 dot (B_08 - B_04) / (B_08 + 2.4 dot B_04 + 1) $
$ "SAVI" = 1.5 dot (B_08 - B_04) / (B_08 + B_04 + 0.5) $
$ "GNDVI" = (B_08 - B_03) / (B_08 + B_03) $

NDVI and EVI2 measure vegetation vigor.
SAVI reduces bare-soil background influence on the vegetation signal.
GNDVI is more sensitive to chlorophyll concentration than NDVI.

=== Red-edge indices

$ "NDRE1" = (B_08 - B_05) / (B_08 + B_05) $
$ "NDRE2" = (B_08 - B_06) / (B_08 + B_06) $
$ "IRECI" = (B_07 - B_04) / (B_05 / (B_06 + epsilon)) $
$ "CRI1" = 1 / B_03 - 1 / B_05 $

NDRE1/NDRE2 separate shrubland from grassland where NDVI saturates.
IRECI measures chlorophyll content using the red-edge inflection.
CRI1 detects carotenoid concentration (vegetation stress).

=== Water and surface indices

$ "NDWI" = (B_03 - B_08) / (B_03 + B_08) $
$ "MNDWI" = (B_03 - B_11) / (B_03 + B_11) $
$ "NDBI" = (B_11 - B_08) / (B_11 + B_08) $
$ "NDMI" = (B_08 - B_11) / (B_08 + B_11) $
$ "NBR" = (B_08 - B_12) / (B_08 + B_12) $
$ "BSI" = ((B_11 + B_04) - (B_08 + B_02)) / ((B_11 + B_04) + (B_08 + B_02)) $
$ "NDTI" = (B_11 - B_12) / (B_11 + B_12) $

NDWI and MNDWI detect open water (MNDWI is better in urban areas).
NDBI highlights built-up/impervious surfaces.
NDMI and NBR capture vegetation moisture and burn scars.
BSI separates bare or sparsely vegetated ground.
NDTI distinguishes crop residue and tillage from naturally bare ground.

== Tasseled Cap transformation (deployed)

The Nedkov @nedkov2017 coefficients project the 10-band reflectance into three axes:
$ T_k = sum_(i=1)^(10) c_(k,i) dot B_i , \ quad k in {"Brightness", "Greenness", "Wetness"} $
Per cell, mean and standard deviation of each component are stored (6 features).

== Spatial statistics (deployed, Model 2 only)

- *Sobel edge magnitude*: 3×3 Sobel filter on NIR → mean, std, max (3 features)
- *Laplacian*: 3×3 Laplacian on NIR → mean absolute value, std (2 features)
- *Moran's I* on NIR: spatial autocorrelation with 4-neighbor weights:
$ I = N / W dot (sum_(i tilde j) z_i z_j) / (sum_i z_i^2) , quad z_i = x_i - overline(x) $
where $N$ is the count of valid pixels, $W$ is the count of valid neighbor pairs, and $i tilde j$ denotes horizontal/vertical adjacency (1 feature)
- *NDVI intra-cell range* and *IQR* (2 features)

These features hurt tree models (overfitting to specific landscapes) but improved MLP accuracy.

== Local Binary Patterns (deployed, Model 2 only)

Rotation-invariant uniform LBP computed on five images: NIR, NDVI, EVI2, SWIR1, NDTI.
$ "LBP"(x_c) = sum_(p=0)^(7) s(g_p - g_c) dot 2^p , quad s(x) = cases(1 "if" x >= 0, 0 "otherwise") $
Patterns with ≤2 bit transitions are _uniform_ (bins 0–8 by popcount); all others map to bin 9.
Per band per cell: 10-bin normalized histogram + Shannon entropy ($H = -sum p_b ln p_b$) → 11 × 5 = 55 features/season.

== SAR features (deployed, Model 2 only)

Sentinel-1 C-band SAR features per season: VV and VH backscatter, cross-polarization ratio $"CR" = "VV" / "VH"$, and Radar Vegetation Index:
$ "RVI" = (4 dot "VH") / ("VV" + "VH") $
Cross-season SAR features: summer/winter ratios (VV, VH, CR), temporal std (VV, VH, CR), temporal CV (VV, VH).

SAR responds to surface roughness and structure rather than colour, providing indirect texture information that is natively compatible with tree splits.

== Temporal and phenological features (deployed)

=== Intra-annual differences

For index $I$, seasons $s_1, s_2$ in year $y$:
$ Delta I^"intra"_(s_1 arrow s_2, y) = I_(s_2, y) - I_(s_1, y) $
Computed for (spring→summer) and (summer→autumn) for all 9 indices, both years (Model 1: 36 features).

=== Inter-annual differences

$ Delta I^"inter"_(s) = I_(s, 2021) - I_(s, 2020) $
For all 9 indices and 3 seasons (Model 1: 27 features).

=== Growing-season range

$ R_(I, y) = I_("autumn", y) - I_("spring", y) $
For NDVI, NDWI, EVI2, BSI per year (Model 1: 8 features).

=== Phenological descriptors (Model 2)

Derived from the seasonal trajectory (spring → summer → autumn) for each index and SAR channel:
- *Amplitude*: max − min of the three seasonal values
- *Peak season*: argmax (encoded as 0/1/2)
- *Slope*: linear trend across seasons
- *Curvature*: second-order difference (concavity of the seasonal arc)

These capture crop phenology, deciduous leaf cycles, and seasonal flooding patterns.

== Features tried and rejected

The following feature groups were implemented and tested but excluded from the final models because they either hurt accuracy (tree-model overfitting to landscape-specific patterns) or provided no measurable improvement over the deployed features.

=== GLCM texture

Gray-Level Co-occurrence Matrix computed on NIR and NDVI (quantized to 32 levels), with distances $d = 1$ and angles $0, pi/4, pi/2, 3pi/4$.
Five Haralick properties extracted per image: contrast, homogeneity, energy, correlation, dissimilarity.
10 features per season per cell.
_Rejected_: no accuracy improvement for either tree or MLP models, and the 32-level quantization on a 10×10 pixel patch produced unreliable statistics.

=== Gabor wavelets

Bank of 12 Gabor filters (3 scales $sigma in {1, 2, 4}$ × 4 orientations $theta in {0°, 45°, 90°, 135°}$, frequency 0.3) applied to normalized NIR.
The frequency parameter (0.3 cycles per pixel) follows scikit-image's convention.
We tried both the real-part response and the full complex-valued response (magnitude and phase), extracting mean and std of each per cell → up to 48 features per season.
_Rejected_: no improvement with either variant; on a 10×10 patch the receptive field exceeds the cell, making filter responses dominated by edge effects.

=== HOG (Histogram of Oriented Gradients)

8-orientation HOG with 5×5 pixel cells and 1×1 block normalization on NIR.
Produces a feature vector of length 32 plus mean and std → 34 features per season.
_Rejected_: the 10×10 patch is too small for meaningful gradient histograms. HOG is designed for object detection in larger image regions.

=== Morphological profiles

Opening, closing, and morphological gradient (closing − opening) on NDVI at disk radii $r in {1, 2, 3}$.
9 features per season.
_Rejected_: collapsed to near-constant values on 10×10 patches. Morphological profiles require larger spatial context to capture structure.

=== Semivariogram

Empirical semivariance at lags 1–4 on NIR, plus exponential model fit ($"nugget" + "sill" dot (1 - e^(-h/"range"))$).
7 features per season (4 gamma values + 3 fit parameters).
_Rejected_: unstable fits on small patches (only 10×10 = 100 pixels). Range estimates frequently hit the upper bound.

=== OSM (OpenStreetMap) features

Spatial features from OSM vector data: building count, building area fraction, mean building area, road count and total length, dominant landuse type (one-hot encoded), and distance to nearest water body.
~15 features per cell (time-invariant).
_Rejected_: they violate the design goal of a self-contained, globally deployable pipeline. OSM coverage is inconsistent across countries and would require maintaining a separate vector data dependency.


= Models

#todo[THIS IS WHERE THE TABLE ABOUT MODELS GOES]

== Model 1: Pixel-wise label classification global model (Ivan Iachnyk)

This model predicts the ESA WorldCover land-cover class for each individual 10 m pixel, which is then used to render the Nuremberg map at arbitrary resolutions: the dashboard simply aggregates pixel-level predictions into whatever grid the user selects.

=== Model type

We use a CatBoost gradient-boosted decision tree (GBDT) trained with the `MultiClass` loss.
We initially experimented with both LightGBM and a small MLP, but settled on CatBoost for three reasons.
First, trees offer substantially better explainability than neural networks.
Second, CatBoost is currently one of the state-of-the-art GBDT frameworks with clean, well-maintained CUDA support, allowing us to train on GPU without workarounds.
Third, CatBoost builds symmetric (balanced) decision trees, which makes inference significantly faster than the asymmetric trees used by LightGBM or XGBoost. It was an important property for us because the inference pipeline runs predictions over millions of pixels.

The final model uses depth-8 trees, a learning rate of 0.03, L2 regularization of 3.0, and early stopping with a patience of 80 rounds.
Inverse-frequency class weights are used to compensate for label imbalance in the training data.

=== Feature vector

Each pixel is represented by a fixed-length vector of 217 features, constructed from multi-temporal, multi-sensor satellite observations (see @feature-engineering for all formulas and definitions).
The feature vector covers two years (2020, 2021) and three seasons (spring, summer, autumn), giving six temporal slots.

For each of the six time slots, we extract:
- 10 raw *Sentinel-2 L2A surface reflectance* bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (60 features total)
- 9 *spectral indices*: NDVI, NDWI, NDBI, NDMI, NBR, BSI, EVI2, NDRE1, NDRE2 (54 features total)
- 3 *Sentinel-1 SAR* features: VV backscatter, VH backscatter, and the VV/VH ratio (18 features total)

On top of the per-slot features, we compute intra-annual differences (spring→summer, summer→autumn) for all 9 indices in both years (36 features), inter-annual differences (same season, 2021 − 2020) for all 9 indices and 3 seasons (27 features), growing-season range (autumn − spring) for NDVI, NDWI, EVI2, BSI per year (8 features), and SAR temporal diffs for VV and VH (14 features).

The final feature set was selected through extensive experimentation: we trained thousands of CatBoost, scikit-learn HGBR and LightGBM configurations across multiple feature subsets to converge on the combination that yielded the best validation accuracy.
Tree-based models cannot effectively utilize computed texture descriptors like Gabor wavelets or LBP — these features decreased accuracy due to overfitting on specific landscapes (see @feature-engineering).
SAR backscatter, however, captures structural information while being natively compatible with tree splits, and is weather-independent.

=== Spatial and temporal resolution

The model operates at the native Sentinel-2 resolution of 10 m per pixel.
We deliberately chose this pixelwise design rather than the 10×10 cell aggregation (used by Model 2) so that we can produce the Nuremberg land-cover map at any resolution the user selects on the dashboard.
The dashboard resolution slider simply aggregates the per-pixel predictions into the desired grid, which keeps the presentation consistent across zoom levels without requiring separate models or resampling tricks.

The temporal design covers three seasons per year to capture the full growing cycle while avoiding winter months, where cloud cover in Central Europe makes reliable optical composites difficult to produce (see @cloud_cover_diagram).

=== Hold-out strategy

The model is trained on data from 100 European cities spanning a wide range of climates and biomes, from Scandinavian tundra and boreal forest to Mediterranean shrubland and Atlantic bogs.
Up to 150,000 pixels are randomly sampled per city, giving a theoretical maximum of 15,000,000 samples × 217 features (~13 GB in float32); the actual count is lower because some cities contain fewer than 150,000 valid pixels.
Nuremberg itself is excluded from both the training and validation sets so that all predictions shown on the dashboard are genuinely out-of-sample.
15 cities are held out as a dedicated validation set (up to 2,250,000 pixels, ~2 GB), selected to cover the full diversity of landscapes (e.g.\ Munich, Stockholm, Seville, Crete, Iceland Highlands, Camargue wetland, Vojvodina cropland).
This geographic split ensures the model is evaluated on regions it has never seen during training.

=== Evaluation beyond accuracy

In addition to top-1 accuracy, we report per-class precision, recall, and F1-score via scikit-learn's `classification_report`, as well as a full confusion matrix.
These per-class metrics were especially important because the class distribution is heavily imbalanced: tree cover and cropland dominate, while shrubland and bare/sparse vegetation are rare, so some tree configs achieved higher overall accuracy by overpredicting dominant classes at the expense of minority ones.
Reporting only overall accuracy would mask poor performance on minority classes.


The geographic hold-out itself also acts as a form of stress test: the validation cities include biomes and landscapes not well represented in the training set (e.g.\ Icelandic highlands, Hungarian steppe, Mediterranean maquis).
Predicting correctly in these climatically and ecologically distinct regions tests whether the model has learned generalizable spectral–temporal patterns or merely overfitted to Central European landscapes.

Known failure modes — particularly regarding rare land-cover classes and inherited label noise — are discussed in the Limitations section (see @rare-labels).

=== Technical setup

All training and experimentation for this model was done on a single laptop: an ASUS ROG Zephyrus G16 (GA605WI) with the following specifications:
- *CPU*: AMD Ryzen AI 9 HX 370, 12 cores / 24 threads
- *RAM*: 32 GB DDR5
- *GPU*: NVIDIA GeForce RTX 4070 Laptop, 8 GB GDDR6
- *Storage*: 1 TB NVMe SSD

The minimum practical requirements for reproducing the training pipeline are approximately 300 GB of free storage (for raw Sentinel-1/2 imagery, WorldCover tiles, and cached feature matrices across 100+ cities), at least 32 GB of system RAM (feature construction for a single city can peak at ~3 GB, and the concatenated training matrix is several GB), and a GPU with at least 8 GB of VRAM and TensorFloat-32 (TF32) support for CatBoost's GPU training mode.

Training a single CatBoost configuration (3000--4000 trees, depth 8) takes roughly one hour on the RTX 4070.
Given that we swept multiple hyperparameter configurations, feature subsets, and framework comparisons (CatBoost vs.~LightGBM vs.~scikit-learn HistGradientBoosting), the practical experimentation phase for this model took several weeks of real time.


== Model 2: Global deployment MLP (softlabel) (Ivan Iachnyk)

While Model 1 operates at pixel level for Nuremberg only, the global deployment model provides land-cover predictions for *_any_* location worldwide at 100m×100m (10×10 pixel) cell resolution.
This model powers the "Global" tab of the dashboard.
It predicts a full class-probability distribution rather than a single label, since each 100m cell typically contains a mix of land-cover types.

=== Model type and architecture

We use a fully connected Multi-Layer Perceptron (MLP) trained with a soft cross-entropy loss on class-fraction labels.
The deployed architecture is a _TaperedMLP_ with four hidden layers of widths 1024→512→256→64 and GELU activations, totalling approximately 2.5 million parameters.
Each hidden layer consists of a linear projection, batch normalization, GELU activation, and dropout - referred to as a _PlainBlock_.
A small input dropout of 0.3% is applied before the first layer.
The output head is a linear layer followed by log-softmax, producing log-probabilities over 7 land-cover classes.

The architecture was selected through a BOHB (Bayesian Optimization + HyperBand) sweep @falkner2018bohb over 100+ trial configurations.
BOHB combines the principled early termination of HyperBand with a kernel density estimator that guides the search toward promising regions of the hyperparameter space.
We searched over four base architectures (512/256/128/64, 1024/512/256/64, 2048/512/128, 2048/1024/512), three activation functions (GELU, SiLU, Mish), dropout rates, learning rates, weight decay, mixup strength, and label denoising thresholds.
Training budgets ranged from 15 epochs (quick rejection) to 300 epochs (full convergence) with an early-stopping patience of $ceil(5000 / "steps per epoch")$ epochs.

The training procedure uses:
- *AdamW* optimizer with cosine annealing learning-rate schedule (3-epoch linear warmup)
- *Mixup* data augmentation (convex combinations of inputs and labels)
- *Soft cross-entropy* loss (supports fractional class labels directly, since cells contain mixed land cover)
- *Mixed-precision training* (FP16 on CUDA with gradient scaling)
- *Label denoising*: class fractions below 2.1% are zeroed and the remaining fractions renormalized, which reduced noise from negligible sub-pixel classes

=== Feature vector

The feature vector is extracted by the Rust _terrapulse_ binary, which processes 100m cells (10×10 Sentinel-2 pixels) into fixed-length feature vectors.
The deployed model uses *1,764 features per cell* in total (see @feature-engineering for all formulas and definitions).
Per cell per season, the Rust extractor produces 224 raw features organized into five groups:

- *Band statistics* (80/season): 10 bands × 8 statistics (mean, std, min, max, Q25, median, Q75, finite fraction) with 20m bands block-reduced before computing
- *Spectral indices* (75/season): all 15 indices (the 9 from Model 1 plus SAVI, MNDWI, GNDVI, NDTI, IRECI, CRI1), each summarized by 5 statistics (mean, std, Q25, median, Q75)
- *Tasseled Cap* (6/season): Brightness, Greenness, Wetness — mean and std each
- *Spatial statistics* (8/season): Sobel edges, Laplacian, Moran's I on NIR, NDVI range/IQR
- *Multi-band LBP* (55/season): rotation-invariant uniform LBP histograms on NIR, NDVI, EVI2, SWIR1, NDTI

With 2 years × 3 seasons = 6 temporal slots, plus cross-season SAR features (VV, VH, CR, RVI and derived ratios/statistics) and phenological descriptors (amplitude, peak season, slope, curvature), this yields the final 1,764-dimensional input.

Spatial, texture, and phenological features are viable for the MLP because neural networks can learn arbitrary non-linear combinations of such descriptors.
In our experiments, LBP and spatial features hurt tree-based models (overfitting to specific landscape patterns) but improved MLP accuracy (see @feature-engineering).

Features are standardized using a global StandardScaler (fitted on training data) prior to training and inference.

=== Inference pipeline

The trained PyTorch model is exported to ONNX format and embedded in the Rust _terrapulse_ binary.
At inference time, the pipeline runs end-to-end without any Python dependency:

+ *Download*: Sentinel-2 and Sentinel-1 composites are streamed directly from Microsoft Planetary Computer via the STAC API.
+ *Composite*: Per-season cloud-free composites are built using SCL-based cloud masking and first-quartile compositing (same algorithm as described in the data section).
+ *Extract*: The Rust feature extractor computes all 224 features per cell per season in parallel using Rayon.
+ *Scale*: Features are standardized using the saved scaler parameters (mean and scale exported as JSON).
+ *Predict*: The ONNX Runtime session runs inference in chunks of 65,536 cells to manage memory.
+ *Threshold*: Class probabilities below 2.1% are zeroed and the remaining distribution renormalized, matching the label denoising applied during training.
+ *Output*: Results are written to Parquet files and optionally serialized to JSON for the dashboard.

The entire pipeline is compiled into a single static binary, making deployment straightforward: the user only needs the binary, the ONNX model file, the scaler JSON, and the model configuration file.

=== Hold-out strategy

The MLP is trained on 92 European cities, validated on 23 cities (optimized for label-distribution balance with a mean class-fraction gap of only 0.04 percentage points between train and val splits), and tested on 6 held-out cities that were _never_ used in either training or validation.
The training set contains 2,695,600 cells (100m×100m patches) × 1,764 features, totalling approximately 19 GB in float32.

#figure(
  table(
    columns: 4,
    [*City*], [*Country*], [*Cells*], [*Purpose*],
    [Nuremberg], [Germany], [29,946], [Explicitly excluded],
    [Ankara], [Turkey], [57,311], [Downloaded for testing],
    [Sofia], [Bulgaria], [36,330], [Downloaded for testing],
    [Riga], [Latvia], [37,818], [Downloaded for testing],
    [Edinburgh], [Scotland], [29,516], [Downloaded for testing],
    [Palermo], [Sicily], [30,430], [Downloaded for testing],
  ),
  caption: [Test cities for model evaluation. None of these cities appeared in the training or validation splits.],
) <mlp-test-cities>

Five of the six test cities (Ankara, Sofia, Riga, Edinburgh, Palermo) were downloaded specifically for evaluation and span different European climatic and geographic regions.

The full lists of training and validation cities are provided in @mlp-city-lists.

=== Evaluation

The deployed model (#7, T_1024/512/256/64 GELU) was selected because it ranks \#1 at both the 5% and 10% fixed evaluation thresholds across all 6 test cities.
Three metrics are reported:
- *Top-1 accuracy*: the fraction of cells where the argmax predicted class matches the argmax true label
- *R²*: per-class coefficient of determination, computed only on cells where the true class fraction exceeds the threshold (to avoid noise from negligible fractions)
- *Combined*: $0.5 dot "Top-1" + 0.5 dot max(0, R^2)$

#figure(
  table(
    columns: 8,
    [*Model*], [*Nuremberg*], [*Ankara*], [*Sofia*], [*Riga*], [*Edinburgh*], [*Palermo*], [*Mean*],
    [Deploy (\#7)], [0.944], [0.842], [0.867], [0.928], [0.936], [0.893], [*0.902*],
    [\#8], [0.944], [0.831], [0.873], [0.934], [0.937], [0.892], [0.902],
    [\#5], [0.942], [0.837], [0.868], [0.928], [0.937], [0.892], [0.901],
    [\#3], [0.943], [0.840], [0.864], [0.929], [0.934], [0.893], [0.901],
    [V8 baseline], [0.928], [0.785], [0.853], [0.905], [0.915], [0.881], [0.878],
  ),
  caption: [Top-1 accuracy per test city. The deployed model achieves 90.2% mean accuracy, a 2.4 percentage point improvement over the V8 baseline. Models: \#8 = 2048/1024/512 GELU (6.2M params), \#5 = 512/256/128/64 GELU (1.1M), \#3 = 512/256/128/64 SiLU (1.1M).],
) <mlp-top1-results>


At the 5% evaluation threshold, the deployed model achieves a combined score of 0.789 (Top-1: 90.2%, R²: 0.676), ranking first among all 10 BOHB candidates and outperforming the V8 baseline (a prior model before BOHB tuning) by 6.6 percentage points on the combined metric.

A per-class R² analysis on Riga (the city with the most diverse class distribution among the test set) reveals that the model achieves strong R² values for tree cover (0.93), water (0.97), built-up (0.91), grassland (0.83), and cropland (0.75), but struggles with shrubland — which has only 75 cells above 1% fraction, making reliable regression effectively impossible.
This mirrors the rare-class difficulty described in @rare-labels.

=== Stress testing <stress-testing>

To assess robustness, three perturbation experiments were conducted on the deployed model across all 6 held-out test cities (221,351 cells).
Perturbations are applied in z-score space (post-StandardScaler); zeroing a feature is equivalent to mean imputation.
Statistical significance is assessed via paired two-sided $t$-tests across the 6 cities.

==== Gaussian noise injection

Additive Gaussian noise $cal(N)(0, sigma^2)$ is injected into the standardized feature vector, with each $sigma$ level averaged over 10 random seeds.
The model degrades gracefully up to $sigma = 0.2$ (R² from 0.676 to 0.653, $p = 0.001$), then rapidly: at $sigma = 1.0$ R² falls to 0.280 and at $sigma = 2.0$ all metrics collapse (R² $= -0.24$, Top-1 $= 65%$, Top-3 $= 27%$).
Full results with 95% confidence intervals are listed in @stress-noise-table.

==== Season dropout

Zeroing all features from a single year$times$season slot (266 columns each) reveals a clear asymmetry: dropping any 2020 season is not statistically significant ($p > 0.05$), while all three 2021 seasons produce significant degradation (e.g. 2021 summer: R² $= 0.569$, $p = 0.001$).
This asymmetry is expected: the ground-truth labels are derived from 2021 WorldCover, so the model's internal representations weight 2021 observations more heavily.

Dropping an entire year is catastrophic: removing all of 2021 yields R² $= -0.42$ ($p = 0.022$), while removing all of 2020 still halves $R^2$ to 0.31 ($p = 0.025$).
Cross-year same-season dropout lowers R² to $approx 0.43$, confirming that no single season suffices.
Full results are listed in @stress-season-table.

==== Feature-group ablation

Zeroing entire feature categories reveals a clear importance hierarchy (@stress-ablation-inline).
Spectral indices are the most critical: removing all 450 index columns collapses R² to $-0.96$ ($p = 0.044$) and Top-3 from 74.0% to 18.3%.
Phenological features show a disproportionate effect on multi-class ranking: Top-3 drops to 51.5% ($p = 0.002$) despite only a modest Top-1 decline.
LBP features (330 columns, 19% of the feature vector) are not statistically significant ($p = 0.185$); spatial features (12 columns) are likewise insignificant ($p = 0.088$).

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    [*Group*], [*Cols*], [*R²*], [*Top-1*], [*Top-3*], [*MAE*], [*$p$*],
    [baseline], [--], [0.676], [0.902], [0.740], [0.033], [--],
    [Indices], [450], [$-$0.96], [0.639], [0.183], [0.104], [0.044],
    [Bands], [480], [0.081], [0.813], [0.488], [0.077], [0.0004],
    [Phenological], [120], [0.584], [0.874], [0.515], [0.049], [0.002],
    [SAR], [336], [0.634], [0.897], [0.713], [0.040], [0.010],
    [Tasseled Cap], [36], [0.659], [0.894], [0.727], [0.036], [0.131],
    [LBP], [330], [0.671], [0.900], [0.730], [0.035], [0.185],
    [Spatial], [12], [0.672], [0.900], [0.738], [0.034], [0.088],
  ),
  caption: [Feature-group ablation summary, ordered by impact. Groups with $p > 0.05$ (LBP, Spatial, Tasseled Cap) show no statistically significant degradation. Extended tables including multi-group ablations and per-city breakdowns are provided in @stress-test-appendix.],
) <stress-ablation-inline>

==== Per-city geographic analysis

Because the model was trained exclusively on European cities, Ankara (Turkey) serves as a natural out-of-distribution test.
At baseline, Ankara already shows the lowest R² among test cities (0.603 vs.\ 0.756--0.813 for the five EU cities) and lowest Top-1 (84.1% vs.\ 87--94%).
Under perturbation, the gap persists but does not widen disproportionately: removing SAR features, for instance, degrades Ankara R² by 0.005 (negligible) while Riga drops by 0.068, suggesting the model relies on SAR more in northern European landscapes.
Removing phenological features degrades all cities substantially, but Ankara's Top-2 accuracy falls furthest (65.3% to 42.3%), consistent with the importance of temporal vegetation signatures in semi-arid landscapes where spectral contrast between seasons is large.

==== Limitations

Stress testing measures robustness to feature removal, not optimality of the feature set.
Although LBP and spatial features individually show no statistically significant contribution when ablated, this does not imply they are unnecessary: all feature groups jointly contribute to the deployed model's peak composite performance (R² $= 0.676$, Top-1 $= 90.2%$), and their marginal gains may be masked by correlations with other groups.
Removing them would reduce the feature vector by 19% but risks degradation under distribution shift not captured by the current test cities.

== Model 3: label change prediction


Model 3 is the first of TWO models used in a two-step prediction process for predicting the labels of future years.

The first step involves a model that predicts how likely a particular cell is to change within the next year (Model 3)
The second model then predicts the new label of cells with a high likelihood of change (Model 4)
Model 3 is a binary random forest with a depth of 35 and 50 estimators, which predicts the likelihood of change in a given cell.

The model trained is being kept simple on purpose, using only 17 features per pixel.
#todo[add why it is being kept simple on purpose (small sentence)]
First, the raw reflectance from Sentinel-2 is used.
Bands 2–8, 8A, 11 and 12 are used as features, the most important of which are:

Band 02: 490 nm (blue), useful for detecting water

Band 03: 560 nm (green), useful for detecting vegetation

Band 04: 665 nm (red), useful for detecting chlorophyll (chlorophyll absorbs red light, so the combination of reflected green light and absorbed red light verifies the detection of plants)

Band 08: 842 nm (near infrared), useful for biomass detection, e.g. forest.

In addition, we calculate the Normalized Difference Vegetation Index (NDVI). This also helps to distinguish vegetation from other things.
#todo[The indices were explained earlier, we can probably just list them here without further explanation]

We also calculate the standard deviation for the NDVI. This provides information about how much vegetation changes, which can indicate vegetation being turned into buildings or forest being turned into cropland.

In addition to the satellite data, we also include the current land use classification and a few more contextual and socioeconomic features, such as population density, the number of residential units, commercial usable space, and the number of cars per 1,000 inhabitants (from @nuremberg_district_statistics), to distinguish residential from industrial areas.


A random forest was chosen for training because it allows for balanced training, which is critical for change detection. It is also a lightweight model that performs well on binary tasks.
#todo[what does "balanced training" mean?]
A resolution of 10 m is necessary to detect small changes in the environment.
As the model is trained on a relatively small area, it is possible to compute it pixel-wise.
The hyperparameters of this model were determined using HPO.
#todo[Which Hyperparameter-optimization technique was used?]
#todo[Maybe just add reference to section at the start where you explain the HPO (which approach, which features and which options for these features)]
As we only have labels for two years, it is difficult to evaluate the model effectively without accidentally leaked information.
Therefore, we decided on a spatial validation strategy.
We use 4-fold partial cross-validation to split Nuremberg into four horizontal strips.
#todo[add more context: "Only for Nuremberg" because we also use socioeconomic data which is only available there]
We then train on three of these strips and test on the remaining strip, which allows the model to train and predict on whole neighborhoods and prevents data leakage.

Because of the fact that only very few pixel change from 2020 to 2021 we did not evaluate the model on it accuracy across the whole test-set. as in this case, simply predicting no change would achieve an accuracy of nearly 95 %.
That is why for evaluating the model we used a test set that contained equally amounts of changing and non changing pixels.
Using this strategy, we achieved an accuracy of 
#todo[add exact accuracy here]
#strike[On this we calculated how many Pixels got classified false what was a little bit over 20 %.]


== Model 4: label next year prediction

This model is the second step of the two-step process, that predicts the labels of the following year.
In the first step the likelihood of a change is predicted and in this second step the new label for this cell is predicted.
For this the same features as in the last models are used, but the prediction is only applied to the cells with a change likelihood of more then 0.95.
This Threshold was obtained by HPO.
#todo[What HPO did we use? how was it used]


Just as the first model this second step is also a random Forest with an depth of 26 and 50 estimators. Those Hyperparameters again were obtained via HPO. The train and test set of this model only contains cells over the change-threshold of 0.95 with the same holdout strategy as in the first model.
#todo[Wait, are both models random forests or gradient boosted decision tree? (which consists of many decision trees itself)]

For this model we get an accuracy of almost 90 % and a F1 Score of almost 70 %.
#todo[Add on which data we get these metrics (probably the spatial holdout cross validation)]

The two stage model overall struggles the most with Bare/Sparse due to very little representation of this class in Nuremberg. Also most of the cells classified as such in 2020 got a different label in
2021 what also could be due to the change in algorithm use for classification by ESA. Also the runway of the airport gets a high probability of change by the model. Most likely because the dark runway has spectral similarities to dug-up ground at a construction site.


= Explainability & Trust
The user can see the land cover change between two years by using the predictions tab of the Nuremberg view. The remaining colored pixels after the years selection contain the classification of the comparison year.
The following paragraphs of this chapter outline why it is important to interpret the predictions carefully and ideally combine them with domain knowledge.

In the predictions tab, the model classifies parts of the airport runway as water in 2021. This is not real water, but more like a visual artifact caused by aircraft activity.
When comparing 2020 and 2021, you may also notice changes in the surrounding grass area of the runway. Despite a visible change in the appearance on the satellite imagery, this doesn't represent actual land cover change and can be misleading if interpreted as such.
At the same time, the model captures changes near the area of Cube One (building of UTN) from 2020 to 2021, which aligns with real construction activity (Hmm, we wonder what is being built there...).
We also find correct explanations/predictions of building activity for other construction projects such as a new development area in the north of "Kornburg" (Street: "Rieterbogen")

A similar pattern appears in the experimental tab. The areas near the runway show a high change probability despite no real change is expected within that area and the area around Cube One shows a high change probability as well which matches the ongoing development.

#todo[Add, which models we are talking about here (model 1, 2, 3, 4, etc.)]


= Limitations and Data Issues

== General Limitations

The overall biggest limitation is posed by using the ESA WorldCover labels as ground-truth.
As outlined in @WorldCover_PUM_v2, these labels themselves are a products of machine learning models themselves.
More specifically, the authors used a Catboost GBDT model trained on features mainly derived from Sentinel 2 data.
The model training for the was based on 260 thousand (2020) and 319 thousand (2021) samples, each consisting of 115 features.
Overall, the accuracy reported by the authors comes to 74.4% for 2020 and 76.7% for 2021.
For our task, this means that predicting the WorldCover labels is nothing more than predicting the output of another ML-model that itself has non-outstanding performance.
We basically train our model to do the same mistakes as the model by Zanaga et al..

== Change Prediction

A quite big limitation arises from the fact that there were different models involved for predicting the WorldCover labels in 2020 as well as 2021.
Especially for the task of change-prediction, our model risks to learn the difference in prediction model rather than the real changes in ground usage.

The reference timeframe is also a limiting factor.
The only "ground truth" labels we have are from 2020 and 2021, which is why this period is the only period we can use for the prediction of change and future labels.
As a result, we have a high possibility of overfitting to the 2020 $arrow$ 2021 change.
A problem which is possibly exacerbated by the fact that construction progress in these years has been heavily influenced by the COVID-19 pandemic.

== Data Issues

=== Label noise due to ambiguous base problem

The problem of land use classification itself is a highly ambiguous, even for humans.
This is especially true when mainly derived from satellite imagery.
The user manual itself (@WorldCover_PUM_v2) acknowledges this e.g. for confusion between urban and bare areas (relevant for Nuremberg) or mangroves and trees (not that relevant for Nuremberg).
If the label is not even obvious for an expert human labeler, then ML-models will also have a hard time fitting to it because the hardest and most ambiguous cases are not only hard for the model to infer but also have a higher likelihood of being mislabeled.
This is only made worse by the fact that these issues get propagated through the ML-model trained for the WorldCover release.

We do not explicitly correct this label noise, because without a more reliable ground-truth dataset or manual relabeling, this correction would be highly speculative and outside the scope of this project.

=== Rare Labels <rare-labels>

Classes like bare/sparse vegetation and shrubland are extremely rare in the Nuremberg area, which makes both training and evaluation unreliable for them.
To assess how trustworthy the ESA WorldCover labels actually are, we compared the 2021 ESA labels against LUCAS 2022 @lucas-2022 in-situ survey points within the Nuremberg region.
For rare classes — bareland in particular — the disagreement rate exceeded 50%: more than half of the LUCAS ground observations did not match the corresponding ESA label.
Even for the more common classes like grassland and cropland, there was notable disagreement between the two sources.
While part of this mismatch could be attributed to the one-year difference between the datasets (2021 ESA vs.~2022 LUCAS), the sheer volume of disagreements across all classes suggests that the discrepancies are not purely temporal.
Shrubland does not even occur within the Nuremberg city boundary in the ESA product, making it effectively impossible to evaluate locally.

=== Seasonal changes regarding cropland

While training the first proof-of-concept models, we quickly realized that cropland in particular changes drastically through the year.
At the start of the year, harvested or freshly plowed fields look like bare ground with no vegetation; by summer, they are fully grown; and common crops like the bright-yellow rapeseed fields around Nuremberg look nothing like typical green vegetation.
Farmers may also leave fields unplanted for soil recovery in some years (German: "Brache", English: "fallow"), making them indistinguishable from bare land on a single-date image.

We addressed this by designing the entire feature pipeline around multi-season composites.
For each year, we construct three separate cloud-free composites (described here @declouding-description) - spring (April–May), summer (June–August), and autumn (September–October) - by downloading all available Sentinel-2 scenes within each window and compositing them per pixel.

#todo[Maybe we can find a single spot to explain the declouding algorithm where we also include the citation @declouding-algorithm]

The key to distinguishing cropland from bare land lies in the seasonal trajectory of vegetation-sensitive indices (see @feature-engineering for all formulas).
In both models, NDVI, EVI2, and BSI are the most relevant indices for this task.
Cropland exhibits a strong seasonal NDVI/EVI2 signal - low in spring (bare soil), high in summer (peak biomass), dropping again in autumn (harvest).
Conversely, BSI behaves inversely: high when soil is exposed (spring/post-harvest), low when vegetation covers the field.
The intra-annual difference features capture this arc directly: for cropland, the spring→summer NDVI difference is large and positive, while for genuinely bare land or built-up areas it stays near zero.

The global MLP model additionally uses SAVI (soil-adjusted, more stable in mixed crop–soil pixels) and NDTI (sensitive to crop residue and tillage) to further aid this separation, along with the Tasseled Cap Greenness component.

Without these multi-season features, any model trained on a single-date composite would systematically confuse spring cropland with bareland, which is exactly the failure mode we observed in our early experiments.

=== Cloud Cover
<declouding-description>

While our solution does not use the cloudless quarterly mosaics provided on #link("https://dataspace.copernicus.eu")[dataspace.copernicus.eu] because we don't want to force our users to set up an account and oauth-access, we apply the same declouding algorithm (outlined in @declouding-algorithm).
This approach is percentile-based and therefore dependent on the availability of sufficiently clear images.
The algorithm is based on the Sentinel-2 L2A scene classification band which flags saturated, cloud shadow, cloud, and thin cirrus pixels) and also tries to somewhat align overall image brightness by taking the first quartile of the stack of observations for each pixel (by reflectance).
However, if only heavily cloud-covered images are available, the method cannot compensate for the lack of usable data.
This problem is especially relevant during the winter months, where cloud cover is frequent.


#figure(
  image("images/cloud_cover_diagram.png", width: 100%),
  caption: [
    Cloud cover diagram of Nuremberg Airport, #link("https://weatherspark.com/y/148228/Average-Weather-at-Nuremberg-Airport-Bavaria-Germany-Year-Round#Figures-CloudCover")[WeatherSpark] @nuremberg-cloud-stats
  ],
) <cloud_cover_diagram>


As illustrated in #ref(<cloud_cover_diagram>), WeatherSpark @nuremberg-cloud-stats states that 5.9 months can be categorized as cloudier part of the year and that December is the cloudiest month during which on average the sky is overcast (80-100% cloud coverage) or mostly cloudy (60-80% cloud coverage) for 72% of the time at the Nuremberg Airport.

== Decision making limitations

The sections above show that as any current ML-based product, our models can *and will* fail sometimes. 
This is why it can and should not be used for automatic (no-human-involved) decisions.

One such decision is based around farmers being banned from converting permanent grassland to arable land (under certain conditions).
While our models could predict this change and could be used by authorities to scan for possible violations, just the model outputs alone should never be used to issue fines or other legal action.
Put plainly, the product could be helpful in identifying *possible* violations of the regulation but needs thorough human involvement, on-site inspections and final decision-making.

= Generative AI Reflection

Below you will find two concrete cases where we disagreed with modelling decisions by ChatGPT.
The Screenshots in the #link(<chatgpt-chat-logs>)[Appendix] show the original prompts and responses.
The prompts were the first prompts of the respective conversations.

== Arguing against ChatGPT - Case 1


Source: Screenshot #link(<chatgpt-chat-log1>)[Chat Log 1] in appendix.

The chat comes from the early stages of the project when a team member asked about the prediction of the *change-likelihood* given the limited data we have (Model 3).

ChatGPT answered that we should be using both 2020 and 2021 satellite image data in the feature vector to predict the change percentage, which we don't think is aligned to the actual goal of what the pipeline around Model 3 is supposed to accomplish

Our main reason for this was that this setup uses information from the target year itself.
In other words: the model would already see the later satellite image when trying to predict whether change happened between 2020 and 2021.
That may improve metrics like accuracy, but it does not match the actual goal of our project.

For us, the important question was not only whether we can detect change afterwards, but whether we can say something about future development.
If 2021 imagery is already part of the input, then the task becomes much closer to retrospective change detection than real prediction, something we think can be done better and more accurately by comparing maps of the relevant timestamps and / or using records of construction sites or by simply comparing the normal land cover predictions of two different years.

This is why we think the proposal is problematic and chose a setup that only uses information which would realistically be available at prediction time.

Note: We do use intra-year features for Model 1 which only does static label-prediction and thus can use "all past data" without the described consistency issues. 
These issues only arise for the *change-likelihood prediction*.


== Arguing against ChatGPT - Case 2

Source: Screenshot  #link(<chatgpt-chat-log2>)[Chat Log 2] in appendix.

In this case, the team member tasked with embedding a resolution slider for the maps on the dashboard asked about how best to implement the prediction back-end for this.

At first we thought that we may need to train models for different resolutions, which is why we fed it into the prompt.
The main point we disagreed on however was the usage of downsampling.
At the time, we already had a model running that predicted the class-makeup of a 10x10 pixel area as percentages (soft labels) and ChatGPT recommended the use of upsampling for more coarse resolutions (which we think is valid) as well as downsampling methods for finer resolutions.
The last part is what we disagree with heavily.
Upsampling does not create new information, it only makes coarse information look finer than it really is.
This a) makes the dashboard less consistent and b) would be very confusing for non-technical users who we would have to explain the method to avoid a misleading presentation of the data.

What we did agree with in the answer was the general idea of aggregation.
As stated earlier in the report, we decided to use a pixelwise model in order to aggregate the predictions for arbitrary resolutions.
So while a valid solution was present in the AI-answer (using aggregation), ChatGPT completely missed the opportunity to actually make the implementation of the slider consistent which it could have done quite easily by simply proposing a pixelwise model.


// References
#bibliography("refs.bib")

#pagebreak()
#counter(page).update(1)
#set page(
  columns: 1,
  footer: context [
    #set align(center)
    Appendix,
    p. #counter(page).display()
  ],
  numbering: "1",
)



= Appendix

== MLP Global Model — City Lists <mlp-city-lists>

=== Training cities (92)

Bremen, Hamburg, Düsseldorf, Leipzig, Amsterdam, Hambach Mine, Welzow Mine, Salzburg, Malmö, London, Brussels, Vienna, Zurich, Munich North, Stuttgart, Innsbruck, Kraków, Budapest, Bratislava, Copenhagen, Gothenburg, Barcelona, Lisbon, Rome, Milan, Lyon, Toulouse, Athens, Almería Coast, Central Hungary, Finnish Lakeland, Swedish Forest, Scottish Highlands, Sicily Interior, Carpathian Romania, Danish Farmland, Dublin, Naples, Valencia, Oslo, Gdańsk, Castilla Meseta, Extremadura Dehesa, Aragón Steppe, Murcia Drylands, Tabernas Desert, Bardenas Reales, Sardinia Maquis, Crete Phrygana, Thessaly Scrubland, Thrace Steppe, El Ejido Greenhouses, Skåne Fields, Trøndelag Farmland, Latvian Farmland, Lithuanian Lowland, Finnish Coastal Farm, Lapland Tundra, Galicia Pastures, Brittany Bocage, Wales Upland, Les Landes Forest, Hortobágy Puszta, Wallachian Steppe, Thracian Farmland, Camargue Wetland, Wadden Tidal, Danube Delta, Pyrénées Meadows, Norwegian Fjord, Carpathian Alpine, Swiss Alps High, Foggia Wheat, Jutland Farmland, Doñana Marshes, Finnish Bog, Mecklenburg Lakes, Danube Floodplain, Cretan Coast, Cyprus (Troodos), Dalmatian Coast, Greek Maquis, Schwarzwald Edge, Algarve Coast, Central Finland Bog, SW Ireland Heath, Andalusia Sierra, Munich, Warsaw, Prague, Seville, Stockholm.

=== Validation cities (23)

Rostock, Paris South, Berlin, Helsinki, Madrid, Alentejo (Portugal), Peloponnese Rural, Po Valley Rural, Dutch Polders, Marseille, Bordeaux, Corsica Interior, Estonian Plains, Iceland Highlands, Ireland Bog Pasture, Vojvodina Cropland, Jaén Olives, Ebro Delta, Andalusia Olives, Central Spain Plateau, Uppland Farmland, Northern Sweden, Dresden.

=== Test cities (6)

Nuremberg (Germany), Ankara (Turkey), Sofia (Bulgaria), Riga (Latvia), Edinburgh (Scotland), Palermo (Sicily).

#pagebreak()
== ChatGPT Chat Logs <chatgpt-chat-logs>

=== Case 1 <chatgpt-chat-log1>


#figure(
  image("images/gpt_disagreement_1_1.png", width: 70%),
  caption: [
    Screenshot of ChatGPT arguing for also using 2021 satellite image data for the label change prediction
  ],
)



=== Case 2 <chatgpt-chat-log2>

#figure(
  image("images/gpt_disagreement_2_1.png", width: 70%),
  caption: [
    Screenshot of ChatGPT recommending to upsample our predictions for finer resolutions Pt.1
  ],
)

#figure(
  image("images/gpt_disagreement_2_2.png", width: 70%),
  caption: [
    Screenshot of ChatGPT recommending to upsample our predictions for finer resolutions Pt.2
  ],
)

= Appendix

== Stress test tables <stress-test-appendix>

All experiments evaluate the deployed MLP (trial 77, 2.5M parameters) on 6 held-out test cities (221,351 cells).
Perturbations are applied in z-score space; zeroing corresponds to mean imputation.
$R^2$ is computed per-class with a 5% threshold, then averaged across classes and cities.
Top-$k$ set-match accuracy uses the deployed label threshold of 2.1%.
$p$-values are from paired two-sided $t$-tests across the 6 test cities.

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    [$sigma$], [R²], [$plus.minus$ 95% CI], [Top-1], [Top-2], [Top-3], [MAE],
    [0.00], [0.676], [--], [0.902], [0.774], [0.740], [0.033],
    [0.05], [0.675], [0.0003], [0.901], [0.773], [0.739], [0.034],
    [0.10], [0.670], [0.0006], [0.900], [0.770], [0.734], [0.034],
    [0.20], [0.653], [0.0012], [0.895], [0.755], [0.717], [0.036],
    [0.50], [0.545], [0.0025], [0.861], [0.666], [0.603], [0.047],
    [1.00], [0.280], [0.0027], [0.783], [0.512], [0.426], [0.069],
    [1.50], [0.012], [0.0027], [0.710], [0.417], [0.328], [0.087],
    [2.00], [$-$0.239], [0.0035], [0.650], [0.356], [0.269], [0.101],
  ),
  caption: [Gaussian noise injection ($sigma$ in z-score units, 10 seeds per level). The model is robust up to $sigma = 0.2$; all levels $sigma >= 0.05$ are statistically significant ($p < 0.05$).],
) <stress-noise-table>

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    [Dropped], [Cols], [R²], [Top-1], [Top-2], [Top-3], [MAE], [$p$],
    [baseline], [--], [0.676], [0.902], [0.774], [0.740], [0.033], [--],
    [2020 spring], [266], [0.656], [0.898], [0.758], [0.723], [0.039], [0.150],
    [2020 summer], [266], [0.644], [0.894], [0.760], [0.727], [0.042], [0.062],
    [2020 autumn], [266], [0.634], [0.895], [0.753], [0.709], [0.040], [0.061],
    [2021 spring], [266], [0.596], [0.897], [0.747], [0.716], [0.041], [0.003],
    [2021 summer], [266], [0.569], [0.892], [0.747], [0.690], [0.043], [0.001],
    [2021 autumn], [266], [0.612], [0.897], [0.750], [0.711], [0.041], [0.019],
    [All 2020], [798], [0.307], [0.836], [0.691], [0.654], [0.075], [0.025],
    [All 2021], [798], [$-$0.424], [0.830], [0.619], [0.607], [0.088], [0.022],
    [Both springs], [532], [0.446], [0.882], [0.707], [0.683], [0.052], [0.030],
    [Both summers], [532], [0.439], [0.876], [0.719], [0.661], [0.055], [0.014],
    [Both autumns], [532], [0.425], [0.877], [0.701], [0.649], [0.055], [0.043],
  ),
  caption: [Season dropout: zeroing all features from one or more temporal slots. Individual 2020 seasons are not statistically significant; 2021 seasons are ($p < 0.02$). Removing an entire year is catastrophic.],
) <stress-season-table>

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    [Removed], [Cols], [R²], [Top-1], [Top-2], [Top-3], [MAE], [$p$],
    [baseline], [--], [0.676], [0.902], [0.774], [0.740], [0.033], [--],
    [Bands], [480], [0.081], [0.813], [0.506], [0.488], [0.077], [0.0004],
    [Indices], [450], [$-$0.963], [0.639], [0.524], [0.183], [0.104], [0.044],
    [LBP], [330], [0.671], [0.900], [0.767], [0.730], [0.035], [0.185],
    [Phenological], [120], [0.584], [0.874], [0.659], [0.515], [0.049], [0.002],
    [SAR], [336], [0.634], [0.897], [0.748], [0.713], [0.040], [0.010],
    [Spatial], [12], [0.672], [0.900], [0.773], [0.738], [0.034], [0.088],
    [Tasseled Cap], [36], [0.659], [0.894], [0.751], [0.727], [0.036], [0.131],
    [Bands+Indices], [930], [$-$2.584], [0.482], [0.522], [0.382], [0.139], [0.050],
    [SAR+LBP], [666], [0.625], [0.893], [0.731], [0.680], [0.042], [0.003],
    [Spatial+Pheno+TC], [168], [0.508], [0.843], [0.641], [0.541], [0.056], [0.002],
  ),
  caption: [Feature-group ablation: zeroing all features in a category. Spectral indices are the most critical; LBP ($p = 0.185$) and spatial features ($p = 0.088$) are not statistically significant.],
) <stress-ablation-table>

