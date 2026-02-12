<p align="center">
  <img src="assets/TerraPulse_logo.png" alt="TerraPulse" width="280" />
</p>

# Mapping Urban Change in Nuremberg with Machine Learning

**Machine Learning WT 25/26** — Grabocka, Asano, Frey — UTN

A tabular ML system that predicts land-cover composition and change in Nuremberg using satellite imagery, ESA WorldCover labels, and OpenStreetMap urban context features.

---

## Project Structure

```
project/
├── scripts/
│   └── download_all_data.py   # Downloads all project data (~650 MB)
├── data/                       # .gitignored — all raw + processed data
│   ├── raw/                    # Sentinel-2 composites + OSM features
│   ├── labels/                 # ESA WorldCover ground-truth maps
│   └── processed/              # Engineered features + grid
├── notebooks/                  # Exploratory analysis
├── src/                        # Production pipeline code
├── reports/                    # Technical report + figures
├── assets/                     # Demo video + images
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2. Download All Data

```bash
python scripts/download_all_data.py
```

This downloads **~650 MB** of data in one go (takes ~10-15 min):

| Step | Data | Source | Size |
|------|------|--------|------|
| 1/3 | ESA WorldCover 2020 & 2021 (labels) | AWS S3 (no auth) | ~190 MB |
| 2/3 | Sentinel-2 composites 2020-2025 (imagery) | Microsoft Planetary Computer (no auth) | ~415 MB |
| 3/3 | OpenStreetMap features (buildings, roads, land-use) | OSM Overpass API (no auth) | ~38 MB |

> **No authentication required.** All data sources are public and free.

### Data Details

#### ESA WorldCover (Labels) — `data/labels/`
- 10 m resolution, 11 land-cover classes
- 2020 (algorithm v100) and 2021 (algorithm v200)
- Tile N48E009 covering Nuremberg region
- **Note**: The two years use different algorithm versions — changes may partly reflect algorithmic differences

#### Sentinel-2 L2A (Features) — `data/raw/`
- Cloud-free median composites (June-August, <20% cloud filter)
- 10 spectral bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B05-B07 (Red Edge), B8A (Narrow NIR), B11-B12 (SWIR)
- 10 m resolution, EPSG:32632 (UTM 32N)
- 6 years: 2020, 2021, 2022, 2023, 2024, 2025

#### OpenStreetMap (Bonus Features) — `data/raw/osm/`
- 105,550 building footprints
- 18,996 road segments + 8,083 intersections
- 3,380 land-use zones (residential, industrial, commercial, etc.)
- 1,233 natural features + 241 water bodies

## Data Citations

- **ESA WorldCover**: Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. https://doi.org/10.5281/zenodo.7254221
- **Sentinel-2**: European Space Agency. Copernicus Sentinel-2 data, processed by ESA.
- **OpenStreetMap**: OpenStreetMap contributors. https://www.openstreetmap.org/copyright

## Team

<!-- Add team members here -->

## License

This project is for academic purposes (UTN Machine Learning WT 25/26).
