#import "ieee_layout.typ": ieee

#show link: underline

#show: ieee.with(
  title: [🌍 TerraPulse - Final Project Report WS25/26],
  abstract: [
    We be trollin', they hatin'.
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



TODO Ivan, Clemens, Robin, Claudius: Code / Repo cleanup?!


= Introduction

This report outlines _TerraPulse_, our machine learning-based application for predicting land-cover composition and land-cover change, with a particular focus on Nuremberg, while also enabling global predictions.

The ESA WorldCover datasets of the years 2020 @worldcover2020 and 2021 @worldcover2021 are used for training multiple models in order to predict:
+ land-cover classification labels
+ change likelihoods
+ new land-cover classification labels

As a result, we´re able to derive the land-cover composition of a satellite image and provide the land-cover change when comparing label classification from multiple satellite images of different years. We even experimented with predicting future label changes within Nuremberg.

In general, sectors like urban planning, environmental monitoring, climate policy as well as business decisions rely on land-cover data. As we've been instructed to focus on Nuremberg, the dashboard shows in the first tab the map of Nuremberg with its districts to provide easy access via the city´s structure level. _TerraPulse_ is also valuable for users outside of Nuremberg as the second tab "Global" allows to select a desired region worldwide, run the classification pipeline for multiple years and evaluate the land-cover classification labels as well.


= Data

All data used in this project is publicly available and does not require any pre-authorized accounts or OAuth credentials, which was a deliberate choice so the inference pipeline can run fully autonomously on any machine.

TODO Clemens: Update "== Data Sources" below (with public link for proper citation)

== Data Sources

*Sentinel-2 Level-2A* surface reflectance imagery is the primary input for both the classification and change-prediction models.
We query it through the Microsoft Planetary Computer STAC API @planetary-computer, which provides free, anonymous access to the full Copernicus Sentinel archive.
Per scene we download bands B02--B08, B8A, B11, B12 (10--20 m resolution) and the L2A Scene Classification Layer (SCL) used by our declouding pipeline.
Scenes are searched per season (spring, summer, autumn) with a cloud-cover ramp (40%→50%→60%) and a ±14-day date expansion fallback when a season has too few usable acquisitions.

*Sentinel-1 GRD (IW mode)* C-band SAR backscatter complements the optical data, particularly for situations where persistent cloud cover makes optical composites unreliable.
Like Sentinel-2, it is accessed via the Planetary Computer STAC API @planetary-computer with no additional credentials.
We prefer ascending orbit scenes for consistency, falling back to any orbit when fewer than three ascending scenes are available within the seasonal window.

*ESA WorldCover 10 m* land-cover maps for 2020 (v100) @worldcover2020 and 2021 (v200) @worldcover2021 serve as our ground-truth labels.
The GeoTIFF tiles are downloaded directly from the public ESA S3 bucket (#link("https://esa-worldcover.s3.eu-central-1.amazonaws.com")[esa-worldcover.s3.eu-central-1.amazonaws.com]).
We map the original 11 ESA classes to a reduced set of 7: tree cover, shrubland, grassland (merging with herbaceous wetland), cropland, built-up, bare/sparse vegetation, and water.

*LUCAS 2022 Survey* @lucas-2022 point observations were used for manual cross-checking of ESA WorldCover labels in ambiguous cases (see @rare-labels below).

*Nuremberg District Statistics* by the City of Nuremberg, accessed 04.03.2026: #link("https://online-service2.nuernberg.de/geoinf/ia_bezirksatlas/atlas.html")

We did not limit ourselves to specific land-cover classes and kept the 10 m pixel grid as spatial unit, aggregating 10×10 pixel patches into cells for the classification model.



= Models

== Model 1: Pixel-wise label classification global model (Ivan Iachnyk)

This model predicts the ESA WorldCover land-cover class for each individual 10 m pixel, which is then used to render the Nuremberg map at arbitrary resolutions: the dashboard simply aggregates pixel-level predictions into whatever grid the user selects.

=== Model type

We use a CatBoost gradient-boosted decision tree (GBDT) trained with the `MultiClass` loss.
We initially experimented with both LightGBM and a small MLP, but settled on CatBoost for three reasons.
First, trees offer substantially better explainability than neural networks.
Second, CatBoost is currently one of the state-of-the-art GBDT frameworks with clean, well-maintained CUDA support, allowing us to train on GPU without workarounds.
Third, CatBoost builds symmetric (balanced) decision trees, which makes inference significantly faster than the asymmetric trees used by LightGBM or XGBoost. It was an important property for us because inference pipeline runs predictions over millions of pixels.

The final model uses depth-8 trees, a learning rate of 0.03, L2 regularization of 3.0, and early stopping with a patience of 80 rounds.
Inverse-frequency class weights are used to compensate for label imbalance in the training data.

=== Feature vector

Each pixel is represented by a fixed-length vector of 217 features, constructed from multi-temporal, multi-sensor satellite observations.
The feature vector covers two years (2020, 2021) and three seasons (spring, summer, autumn), giving six temporal slots.

For each of the six time slots, we extract:
- 10 raw *Sentinel-2 L2A surface reflectance* bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (60 features total)
- 9 *spectral indices* computed from these bands (54 features total):

$ "NDVI" = (B_08 - B_04) / (B_08 + B_04) $
$ "NDWI" = (B_03 - B_08) / (B_03 + B_08) $
$ "NDBI" = (B_11 - B_08) / (B_11 + B_08) $
$ "NDMI" = (B_08 - B_11) / (B_08 + B_11) $
$ "NBR" = (B_08 - B_12) / (B_08 + B_12) $
$ "BSI" = ((B_11 + B_04) - (B_08 + B_02)) / ((B_11 + B_04) + (B_08 + B_02)) $
$ "EVI2" = 2.5 dot (B_08 - B_04) / (B_08 + 2.4 dot B_04 + 1) $
$ "NDRE1" = (B_08 - B_05) / (B_08 + B_05) $
$ "NDRE2" = (B_08 - B_06) / (B_08 + B_06) $

- 3 *Sentinel-1 SAR* features: VV backscatter, VH backscatter, and the VV/VH ratio (18 features total)

On top of the per-slot features, we compute temporal difference features. For any index $I$ and seasons $s_1, s_2$ in year $y$:

$ Delta I^"intra"_(s_1 arrow s_2, y) = I_(s_2, y) - I_(s_1, y) $

computed for (spring→summer) and (summer→autumn), for all 9 indices and both years (36 features).

Inter-annual diffs compare the same season across years:

$ Delta I^"inter"_(s) = I_(s, 2021) - I_(s, 2020) $

for all 9 indices and 3 seasons (27 features).

Growing season range captures the full amplitude:

$ R_(I, y) = I_("autumn", y) - I_("spring", y) $

for NDVI, NDWI, EVI2, and BSI per year (8 features).

SAR temporal diffs follow the same pattern for VV and VH backscatter (14 features).

=== Feature justification

Each spectral index was chosen because it targets a specific land-cover signal.
NDVI and EVI2 respond to vegetation vigor, making them effective at separating tree cover, grassland, and cropland.
NDWI is sensitive to open water surfaces.
NDBI highlights built-up and impervious areas.
NDMI and NBR capture vegetation moisture content and respond to burn scars.
BSI (Bare Soil Index) helps distinguish bare or sparsely vegetated ground from other classes.
NDRE1 and NDRE2 use the red-edge bands (B05, B06) and are more sensitive to subtle differences in vegetation health than NDVI alone, which helps separate shrubland from grassland.

The Sentinel-1 SAR features were included because SAR provides information that is orthogonal to optical reflectance.
C-band backscatter responds to surface roughness and structure rather than colour, giving the model a form of indirect texture information.
In our experiments we found that tree-based models cannot effectively utilize computed texture descriptors like Gabor wavelets or LBP. These features simply decreased the accuracy accuracy because of the overfitting on specific landscapes.
SAR backscatter, however, does capture some of the same structural information while being natively compatible with tree splits.
As a secondary benefit, SAR is weather-independent and complements the optical composites during seasons with heavy cloud cover.

The temporal difference features capture phenological cycles.
Cropland has a strong seasonal NDVI signal (bare in winter, green in summer, harvested in autumn), whereas forest stays relatively stable year-round.
These seasonal amplitude and difference features allow the model to separate classes that look similar in a single snapshot but behave differently over time.

The final feature set was selected through extensive experimentation: we trained thousands of CatBoost, scikit-learn HGBR and LightGBM trees configurations across multiple feature subsets to converge on the combination that yielded the best validation accuracy.

=== Spatial and temporal resolution

The model operates at the native Sentinel-2 resolution of 10 m per pixel.
We deliberately chose this pixelwise design rather than the 10×10 cell aggregation (used by Model 2) so that we can produce the Nuremberg land-cover map at any resolution the user selects on the dashboard.
The dashboard resolution slider simply aggregates the per-pixel predictions into the desired grid, which keeps the presentation consistent across zoom levels without requiring separate models or resampling tricks.

The temporal design covers three seasons per year to capture the full growing cycle while avoiding winter months, where cloud cover in Central Europe makes reliable optical composites difficult to produce (see @cloud_cover_diagram).

=== Hold-out strategy

The model is trained on data from 100 European cities spanning a wide range of climates and biomes, from Scandinavian tundra and boreal forest to Mediterranean shrubland and Atlantic bogs.
Up to 150,000 pixels are randomly sampled per city, giving a maximum training set of 15,000,000 samples × 217 features (~13 GB in float32).
Nuremberg itself is excluded from both the training and validation sets so that all predictions shown on the dashboard are genuinely out-of-sample.
15 cities are held out as a dedicated validation set (up to 2,250,000 pixels, ~2 GB), selected to cover the full diversity of landscapes (e.g.\ Munich, Stockholm, Seville, Crete, Iceland Highlands, Camargue wetland, Vojvodina cropland).
This geographic split ensures the model is evaluated on regions it has never seen during training.

=== Evaluation beyond accuracy

In addition to top-1 accuracy, we report per-class precision, recall, and F1-score via scikit-learn's `classification_report`, as well as a full confusion matrix.
These per-class metrics were especially important because the class distribution is heavily imbalanced: tree cover and cropland dominate, while shrubland and bare/sparse vegetation are rare, so some tree configs were better on general metrics by overpredicting rare classes.
Reporting only overall accuracy would mask poor performance on minority classes.

=== Stress testing

The geographic hold-out itself acts as a form of stress test: the validation cities include biomes and landscapes not well represented in the training set (e.g.\ Icelandic highlands, Hungarian steppe, Mediterranean maquis).
Predicting correctly in these climatically and ecologically distinct regions tests whether the model has learned generalizable spectral–temporal patterns or merely overfitted to Central European landscapes.

=== Where and why the model is likely wrong

The model struggles most with rare land-cover classes: particularly bare/sparse vegetation and shrubland.
These classes are inherently hard to separate from grassland and cropland because their spectral signatures overlap heavily, and even human experts find the distinction ambiguous when looking at satellite imagery alone.
The problem is complicated further by the fact that our ground-truth labels come from ESA WorldCover, which itself reports lower accuracy for these classes.
We cross-checked WorldCover labels against LUCAS 2022 @lucas-2022 in-situ point observations and found that even manual inspection of individual bareland predictions was inconclusive. LUCAS labels disagreed with ESA labels more than 50% of the time.
In other words, the model inherits and reproduces the systematic errors of the ESA product, especially where the ESA model was itself uncertain.

=== Technical setup

All training and experimentation for this model was done on a single laptop: an ASUS ROG Zephyrus G16 (GA605WI) with the following specifications:
- *CPU*: AMD Ryzen AI 9 HX 370, 12 cores / 24 threads
- *RAM*: 32 GB DDR5
- *GPU*: NVIDIA GeForce RTX 4070 Laptop, 8 GB GDDR6
- *Storage*: 1 TB NVMe SSD

The minimum practical requirements for reproducing the training pipeline are approximately 150 GB of free storage (for raw Sentinel-1/2 imagery, WorldCover tiles, and cached feature matrices across 50+ cities), at least 32 GB of system RAM (feature construction for a single city can peak at ~3 GB, and the concatenated training matrix is several GB), and a GPU with at least 8 GB of VRAM and TensorFloat-32 (TF32) support for CatBoost's GPU training mode.

Training a single CatBoost configuration (3000--4000 trees, depth 8) takes roughly one hour on the RTX 4070.
Given that we swept multiple hyperparameter configurations, feature subsets, and framework comparisons (CatBoost vs.~LightGBM vs.~scikit-learn HistGradientBoosting), the practical experimentation phase for this model took several weeks of real time.


== Model 2: Global deployment MLP (Ivan Iachnyk)

While Model 1 operates at pixel level for Nuremberg only, the global deployment model provides land-cover predictions for _any_ location worldwide at 100m×100m (10×10 pixel) cell resolution.
This model powers the "Global" tab of the dashboard.
It predicts a full class-probability distribution rather than a single label, since each 100m cell typically contains a mix of land-cover types.

=== Model type and architecture

We use a fully connected Multi-Layer Perceptron (MLP) trained with a soft cross-entropy loss on class-fraction labels.
The deployed architecture is a _TaperedMLP_ with four hidden layers of widths 1024→512→256→64 and GELU activations, totalling approximately 2.5 million parameters.
Each hidden layer consists of a linear projection, batch normalization, GELU activation, and dropout — referred to as a _PlainBlock_.
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
The deployed model uses *1,764 features per cell* in total.
Per cell per season, the Rust extractor produces 224 raw features organized into five groups described below.
With 2 years × 3 seasons = 6 temporal slots, plus cross-season SAR and phenological features, this yields the final 1,764-dimensional input.

==== Band statistics (80 features/season)

For each of the 10 Sentinel-2 bands (B02–B12), 8 statistics are computed across the 100 pixels in the cell: mean, standard deviation, min, max, Q25, median, Q75, and finite fraction.
20m bands (B05, B06, B07, B8A, B11, B12) are first block-reduced to 5×5 via 2×2 mean pooling before computing statistics.

==== Spectral indices (75 features/season)

15 normalized vegetation/surface indices are computed, each summarized by 5 statistics (mean, std, Q25, median, Q75) across the cell.
The 9 indices shared with Model 1 are:
$ "NDVI" = (B_08 - B_04) / (B_08 + B_04) , quad "NDWI" = (B_03 - B_08) / (B_03 + B_08) $
$ "NDBI" = (B_11 - B_08) / (B_11 + B_08) , quad "NDMI" = (B_08 - B_11) / (B_08 + B_11) $
$ "NBR" = (B_08 - B_12) / (B_08 + B_12) , quad "NDRE1" = (B_08 - B_05) / (B_08 + B_05) $
$ "NDRE2" = (B_08 - B_06) / (B_08 + B_06) $
$ "BSI" = ((B_11 + B_04) - (B_08 + B_02)) / ((B_11 + B_04) + (B_08 + B_02)) $
$ "EVI2" = 2.5 dot (B_08 - B_04) / (B_08 + 2.4 dot B_04 + 1) $

The 6 additional indices used only in Model 2 are:
$ "SAVI" = 1.5 dot (B_08 - B_04) / (B_08 + B_04 + 0.5) $
$ "MNDWI" = (B_03 - B_11) / (B_03 + B_11) , quad "GNDVI" = (B_08 - B_03) / (B_08 + B_03) $
$ "NDTI" = (B_11 - B_12) / (B_11 + B_12) $
$ "IRECI" = (B_07 - B_04) / (B_05 / (B_06 + epsilon)) $
$ "CRI1" = 1 / B_03 - 1 / B_05 $

==== Tasseled Cap (6 features/season)

The Tasseled Cap transformation projects the 10-band reflectance into three interpretable axes using the Nedkov (2017) coefficients:
$ T_k = sum_(i=1)^(10) c_(k,i) dot B_i , quad k in {"Brightness", "Greenness", "Wetness"} $

For each component, mean and standard deviation across the cell are stored (6 features).

==== Spatial statistics (8 features/season)

- *Sobel edge magnitude*: 3×3 Sobel filter on the NaN-filled NIR band, summarized by mean, std, max over the cell (3 features)
- *Laplacian*: 3×3 Laplacian filter on the NIR band, summarized by mean absolute value and std (2 features)
- *Moran's I* on NIR: spatial autocorrelation with 4-neighbor weights (right + down), computed as:
$ I = N / W dot (sum_(i tilde j) z_i z_j) / (sum_i z_i^2) , quad z_i = x_i - overline(x) $
  where $N$ is the count of valid pixels, $W$ is the count of valid neighbor pairs, and $i tilde j$ denotes horizontal or vertical adjacency (1 feature)
- *NDVI intra-cell range* and *IQR* (2 features)

==== Multi-band LBP (55 features/season)

Local Binary Patterns (LBP) are computed as 8-neighbor rotation-invariant uniform patterns on five images: NIR, NDVI, EVI2, SWIR1, and NDTI.
Each pixel receives a code:
$ "LBP"(x_c) = sum_(p=0)^(7) s(g_p - g_c) dot 2^p , quad s(x) = cases(1 "if" x >= 0, 0 "otherwise") $

where $g_c$ is the center pixel intensity and $g_p$ are the 8 bilinearly interpolated neighbors at radius 1.
Patterns with ≤ 2 bit transitions are _uniform_ and mapped to bins 0–8 (by popcount); all others go to a single non-uniform bin (bin 9).
Per band per cell, this yields a 10-bin normalized histogram plus Shannon entropy:
$ H = -sum_(b=0)^(9) p_b ln p_b $
giving 11 features × 5 bands = 55 features per season.

==== SAR features (per year)

Sentinel-1 SAR features include per-season VV and VH backscatter, the cross-polarization ratio $"CR" = "VV" / "VH"$, and the Radar Vegetation Index:
$ "RVI" = (4 dot "VH") / ("VV" + "VH") $

Cross-season SAR features are also computed: summer/winter ratios (VV, VH, CR), temporal standard deviation (VV, VH, CR), and temporal coefficient of variation (VV, VH).

==== Phenological features (per year)

For each spectral index and SAR channel, phenological descriptors are derived from the seasonal trajectory (spring → summer → autumn):
- *Amplitude*: max - min of the seasonal values
- *Peak season*: argmax of the seasonal values (encoded as 0/1/2)
- *Slope*: linear trend across seasons
- *Curvature*: second-order difference (concavity of the seasonal arc)

These capture crop phenology, deciduous forest leaf-on/off cycles, and seasonal flooding patterns that cannot be expressed by per-season features alone.

These spatial, texture, and phenological features are viable for the MLP because, unlike CatBoost's balanced trees, neural networks can learn arbitrary non-linear combinations of such descriptors.
In our experiments, LBP and spatial features hurt tree-based models (overfitting to specific landscape patterns) but improved MLP accuracy.

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
    [Deployed (\#7)], [0.944], [0.842], [0.867], [0.928], [0.936], [0.893], [*0.902*],
    [\#8], [0.944], [0.831], [0.873], [0.934], [0.937], [0.892], [0.902],
    [\#5], [0.942], [0.837], [0.868], [0.928], [0.937], [0.892], [0.901],
    [\#3], [0.943], [0.840], [0.864], [0.929], [0.934], [0.893], [0.901],
    [V8 baseline], [0.928], [0.785], [0.853], [0.905], [0.915], [0.881], [0.878],
  ),
  caption: [Top-1 accuracy per test city. The deployed model achieves 90.2% mean accuracy, a 2.4 percentage point improvement over the V8 baseline. Models: \#8 = 2048/1024/512 GELU (6.2M params), \#5 = 512/256/128/64 GELU (1.1M), \#3 = 512/256/128/64 SiLU (1.1M).],
) <mlp-top1-results>

At the 5% evaluation threshold, the deployed model achieves a combined score of 0.789 (Top-1: 90.2%, R²: 0.676), ranking first among all 10 BOHB candidates and outperforming the V8 baseline by 6.6 percentage points on the combined metric.

A per-class R² analysis on Riga (the city with the most diverse class distribution among the test set) reveals that the model achieves strong R² values for tree cover (0.93), water (0.97), built-up (0.91), grassland (0.83), and cropland (0.75), but struggles with shrubland — which has only 75 cells above 1% fraction, making reliable regression effectively impossible.
This mirrors the rare-class difficulty described in @rare-labels.

== Model 3: label change prediction
TODO Clemens:
How does the tabular or fixed-length feature vector of the model look like?
What features did you engineer / aggregate from imagery?
What is the model type?
How do you justify your feature choices?
How do you justify your model choices?
How do you justify your spatial and temporal resolution?
What spatial or temporal hold-out strategy did you use?
What change-specific metric (e.g. false change rate, stability) can you provide as evaluation beyond accuracy?
What stress test (e.g. feature noise, missing data) can we provide?
Where and why is the model is likely wrong?


A two-step prediction model was developed for predicting the labels of future years.
The first step involves a model that predicts how likely a particular cell is to change within the next year. The second model then predicts the new label of cells with a high likelihood of change.
This model is a binary random forest with a maximum depth of 15, which predicts the likelihood of change in a given cell.
The model is trained using only 17 features per pixel.
First, the raw reflectance from Sentinel-2 is used.
Bands 2–8, 8A, 11 and 12 are used as features, the most important of which are:
Band 02: 490 nm (blue), useful for detecting water
Band 03: 560 nm (green), useful for detecting vegetation
Band 04: 665 nm (red), useful for detecting chlorophyll (chlorophyll absorbs red light, so the combination of reflected green light and absorbed red light verifies the detection of plants)
Band 08: 842 nm (near infrared), useful for biomass detection, e.g. forest.

In addition, we calculate the Normalized Difference Vegetation Index (NDVI). This also helps to distinguish vegetation from other things.

$"NIR - Red"/"NIR + Red"$

We also calculate the standard deviation for the NDVI. This provides information about how much vegetation changes, which can indicate vegetation being turned into buildings or forest being turned into cropland.

In addition to the satellite data, we also include the current land use classification and a few more contextual and socioeconomic features, such as population density, the number of residential units, commercial usable space, and the number of cars per 1,000 inhabitants, to distinguish residential from industrial areas.

A random forest was chosen for training because it allows for balanced training, which is critical for change detection. It is also a lightweight model that performs well on binary tasks.
A resolution of 10 m is necessary to detect small changes in the environment. As this is trained on a relatively small area, it is possible to compute it pixelwise.
As we only have labels for two years, it is difficult to evaluate the model effectively. Therefore, we use 4-fold partial cross-validation to split Nuremberg into four horizontal strips. We then train on three of these strips and test on the remaining strip, which allows the model to train and predict on whole neighbourhoods and prevents data leakage.

#code(
  ```python
  # Code example in Typst
  print(f"Hello world!")
  ```,
)

== Model 3: label next year prediction
TODO Clemens:
How does the tabular or fixed-length feature vector of the model look like?
What features did you engineer / aggregate from imagery?
What is the model type?
How do you justify your feature choices?
How do you justify your model choices?
How do you justify your spatial and temporal resolution?
What spatial or temporal hold-out strategy did you use?
What change-specific metric (e.g. false change rate, stability) can you provide as evaluation beyond accuracy?
What stress test (e.g. feature noise, missing data) can we provide?
Where and why is the model is likely wrong?

#code(
  ```python
  # Code example in Typst
  print(f"Hello world!")
  ```,
)

= Explainability & Trust
TODO: Claudius Robin
Your system must explain to a non-expert:
- What changed
- Where it changed
- How confident the system is
You must show:
- One explanation that is helpful
- One explanation that could be misleading, and why
  -




= Limitations and Data Issues
TODO Ivan, Clemens, Robin, #strike[Claudius]: Our "working interactive system" must have minimum feature "Uncertainty and limitation explanations". I think Uncertainty is already represented on the dashboard. Do we already provide limitation explanations on the dashboard?



== General Limitations

The overall biggest limitation is posed by using the ESA WorldCover labels as ground-truth.
As outlined in @WorldCover_PUM_v2, these labels themselves are a products of machine learning models themselves.
More specifically, the authors used a Catboost GBDT model trained on features mainly derived from Sentinel 2 data.
The model training for the was based on 260 thousand (2020) and 319 thousand (2021) samples, each consisting of 115 features.
Overall, the accuracy reported by the authors comes to 74.4% for 2020 and 76.7% for 2021.
For our task, this means that predicting the WorldCover labels is nothing more than predicting the output of another ML-model that itself has non-outstanding performance.
We basically train our model to do the same mistakes as the model by Zanaga et al..

== Change Prediction

A quite big limitation arises from the fact that there were different models involved for predicting the labels in 2020 as well as 2021.
Especially for the task of change-prediction, our model risks to learn the difference in prediction model rather than the real changes in ground usage.

The reference timeframe is also a limiting factor.
The only "ground truth" labels we have are from 2020 and 2021, which is why this period is the only period we can use for the prediction of change and future labels.
As a result, we have a high possibility of overfitting to the 2020 $arrow$ 2021 change.
A problem which is possibly exacerbated by the fact that construction progress in these years has been heavily influenced by the COVID-19 pandemic.

== Data Issues
TODO Robin: Add
#strike["Identify at least three non-trivial data issues, such as:
  - Seasonal effects
  - Cloud cover and missing data
  - Label noise in land-cover maps
  - Spatial resolution mismatch
  - Spatial autocorrelation
  Explicitly choose one issue you do not fix, and justify why."]


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
For each year, we construct three separate cloud-free composites — spring (April–May), summer (June–August), and autumn (September–October) — by downloading all available Sentinel-2 scenes within each window and compositing them per pixel.
Cloud masking is done using the L2A Scene Classification Layer (SCL), which flags saturated, cloud shadow, cloud, and thin cirrus pixels.
The composite itself uses a first-quartile (Q1) approach rather than a simple median: per-pixel, we sort all valid (cloud-free) observations by reflectance and take the 25th percentile value, which tends to suppress residual haze and brightness inconsistencies across scenes.

The key to distinguishing cropland from bare land lies in the seasonal trajectory of vegetation-sensitive indices.
In both models, the following indices are most relevant for this task:

$ "NDVI" = (B_08 - B_04) / (B_08 + B_04) $
$ "EVI2" = 2.5 dot (B_08 - B_04) / (B_08 + 2.4 dot B_04 + 1) $
$ "BSI" = ((B_11 + B_04) - (B_08 + B_02)) / ((B_11 + B_04) + (B_08 + B_02)) $

Cropland exhibits a strong seasonal NDVI/EVI2 signal — low in spring (bare soil), high in summer (peak biomass), dropping again in autumn (harvest).
Conversely, BSI behaves inversely: high when soil is exposed (spring/post-harvest), low when vegetation covers the field.
The temporal difference features capture this arc directly:

$ Delta "NDVI"^"intra"_("spring" arrow "summer", y) = "NDVI"_("summer", y) - "NDVI"_("spring", y) $

For cropland, this value is large and positive, while for genuinely bare land or built-up areas, it stays near zero.
The growing season range further quantifies this:

$ R_("NDVI", y) = "NDVI"_("autumn", y) - "NDVI"_("spring", y) $

The gloabl MLP model additionally uses indices that further aid this separation:

$ "SAVI" = 1.5 dot (B_08 - B_04) / (B_08 + B_04 + 0.5) $
$ "NDTI" = (B_11 - B_12) / (B_11 + B_12) $

SAVI is a soil-adjusted vegetation index that reduces the influence of bare soil background on the vegetation signal, making it more stable in mixed crop–soil pixels early in the growing season.
NDTI (Normalized Difference Tillage Index) uses both SWIR bands (B11, B12) and is sensitive to crop residue and tillage practices — it helps separate recently harvested fields from naturally bare ground.
Additionally, the MLP's Tasseled Cap Greenness component provides another angle on the same phenological signal.

Without these multi-season features, any model trained on a single-date composite would systematically confuse spring cropland with bareland, which is exactly the failure mode we observed in our early experiments.

=== Cloud Cover

While our solution does not use the cloudless quarterly mosaics provided on #link("https://dataspace.copernicus.eu")[dataspace.copernicus.eu] because we don't want to force our users to set up an account and oauth-access, we apply the same declouding algorithm (outlined in @declouding-algorithm).
This approach is percentile-based and therefore dependent on the availability of sufficiently clear images.
The algorithm is based on the Sentinel-2 L2A scene classification band and also tries to somewhat align overall image brightness by taking the first quartile of the stack of observations for each pixel (by reflectance).
However, if only heavily cloud-covered images are available, the method cannot compensate for the lack of usable data.
This problem is especially relevant during the winter months, where cloud cover is frequent.

#figure(
  image("images/cloud_cover_diagram.png", width: 100%),
  caption: [
    Cloud cover diagram of Nuremberg Airport, #link("https://weatherspark.com/y/148228/Average-Weather-at-Nuremberg-Airport-Bavaria-Germany-Year-Round#Figures-CloudCover")[WeatherSpark] @nuremberg-cloud-stats
  ],
) <cloud_cover_diagram>

#strike[TODO Replaced See Below: As illustrated in #ref(<cloud_cover_diagram>), in Nuremberg the probability of at least 80% cloud cover by area in December is approximately 70%.The probability of at least 60% cloud cover is nearly 85%.
  Under these conditions, it is highly possible that even our declouded images might feature irregularities like cloud shadows or even completely unusable pixels.]

As illustrated in #ref(<cloud_cover_diagram>), WeatherSpark @nuremberg-cloud-stats states that 5.9 months can be categorized as cloudier part of the year and that December is the cloudiest month during which on average the sky is overcast (80-100% cloud coverage) or mostly cloudy (60-80% cloud coverage) for 72% of the time at the Nuremberg Airport.

== Decision making limitations

TODO Robin:
- #text(weight: "bold")[Which decisions must not be made based on your results?]

= Generative AI Reflection
#strike[TODO Robin: On what kind of decision do you disagree and explain why? ]

Below you will find two concrete cases where we disagreed with modelling decisions by ChatGPT.
The Screenshots in the #link(<chatgpt-chat-logs>)[Appendix] show the original prompts and responses.
The prompts were the first prompts of the respective conversations.

== Arguing against ChatGPT - Case 1


Source: Screenshot #link(<chatgpt-chat-log1>)[Chat Log 1] in appendix.

The chat comes from the early stages of the project when a team member asked about the prediction of the change-likelihood given the limited data we have.

ChatGPT answered that we should be using both 2020 and 2021 satellite image data in the feature vector to predict the change percentage, which we don't think is aligned to the actual goal of the described project.

Our main reason for this was that this setup uses information from the target year itself.
In other words: the model would already see the later satellite image when trying to predict whether change happened between 2020 and 2021.
That may improve metrics like accuracy, but it does not match the actual goal of our project.

For us, the important question was not only whether we can detect change afterwards, but whether we can say something about future development.
If 2021 imagery is already part of the input, then the task becomes much closer to retrospective change detection than real prediction, something we think can be done better and more accurately by comparing maps of the relevant timestamps and / or using records of construction sites or by simply comparing the normal land cover predictions of two different years.

This is why we think the proposal is problematic and chose a setup that only uses information which would realistically be available at prediction time.


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

