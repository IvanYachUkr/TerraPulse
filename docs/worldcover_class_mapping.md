# WorldCover Class Mapping

This document describes how ESA WorldCover 10m land cover classes are mapped
to the model's output classes used throughout the TerraPulse pipeline.

## Source

Labels are derived from [ESA WorldCover](https://esa-worldcover.org/)
10m-resolution land cover maps (v1.0 for 2020, v2.0 for 2021).

## Mapping (7 output classes)

| ESA Code | ESA Class          | Our Index | Our Class      | Notes                             |
|:--------:|:-------------------|:---------:|:---------------|:----------------------------------|
| 10       | Tree cover         | **0**     | `tree_cover`   | 30.9% of all pixels               |
| 20       | Shrubland          | **1**     | `shrubland`    | 0.5% — key for Mediterranean      |
| 30       | Grassland          | **2**     | `grassland`    | 13.5%                             |
| 90       | Herbaceous wetland | **2**     | `grassland`    | 0.14% — merged, spectrally similar|
| 40       | Cropland           | **3**     | `cropland`     | 12.0%                             |
| 50       | Built-up           | **4**     | `built_up`     | 32.5% — dominant in urban AOIs    |
| 60       | Bare / sparse      | **5**     | `bare_sparse`  | 1.0%                              |
| 70       | Snow and ice       | **5**     | `bare_sparse`  | 0.004% — negligible, merged       |
| 100      | Moss and lichen    | **5**     | `bare_sparse`  | 0.006% — negligible, merged       |
| 80       | Permanent water    | **6**     | `water`        | 9.5%                              |
| 95       | Mangroves          | —         | *(not present)* | 0% in European cities            |

## Design Rationale

- **Shrubland separated** from grassland because it is ecologically distinct
  and important for Mediterranean cities (Marseille 12.2%, Athens 7.8%, Seville 0.6%).
- **Herbaceous wetland merged** into grassland — only 0.14% of pixels,
  spectrally similar to grassland in Sentinel-2 imagery.
- **Snow/ice and moss/lichen merged** into bare_sparse — combined <0.01%
  of pixels across all 56 European cities.

## Label Generation

Labels are generated per 100 m grid cell (10×10 Sentinel-2 pixels) by:

1. Downloading ESA WorldCover GeoTIFF tiles (3°×3° each)
2. If a city spans multiple tiles, all overlapping tiles are mosaiced
3. Reprojecting to the city's UTM anchor grid (nearest-neighbour)
4. Aggregating the 10×10 pixel patch per cell into class proportions
5. Normalizing each row to sum to 1.0

Output: `labels_{year}.parquet` with columns
`[cell_id, mapped_pixels, coverage, tree_cover, shrubland, grassland, cropland, built_up, bare_sparse, water]`

## Code Reference

Defined in [`run_multi_city_pipeline_v5.py`](scripts/run_multi_city_pipeline_v5.py):

```python
WC_CLASS_MAP = {10: 0, 20: 1, 30: 2, 90: 2, 40: 3, 50: 4,
                60: 5, 70: 5, 100: 5, 80: 6}
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
N_CLASSES = 7
```
