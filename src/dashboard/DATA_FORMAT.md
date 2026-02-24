# Dashboard Binary Data Format

## Overview

The Nuremberg dashboard displays land cover maps as flat binary files (`.bin`). Each file is a raw `uint8` array — **one byte per cell**, no header.

## Class Mapping

| Index | Class | RGB Color |
|-------|-------|-----------|
| 0 | tree_cover | (45, 106, 79) |
| 1 | grassland | (149, 213, 178) |
| 2 | cropland | (244, 162, 97) |
| 3 | built_up | (231, 111, 81) |
| 4 | bare_sparse | (212, 163, 115) |
| 5 | water | (0, 150, 199) |
| 255 | outside boundary | transparent |

> **Note**: Shrubland is remapped to grassland (index 1) for Nuremberg.

## File Layout

- **Format**: raw `uint8`, row-major (top-left → bottom-right)
- **Size**: `width × height` bytes (from metadata JSON)
- **No header** — just pixel values

## Resolutions

Each year needs files at **10 resolutions** (aggregation levels):

| Resolution | Pixel Size | Grid (width × height) | File Size |
|-----------|-----------|----------------------|----------|
| res1 | 10m | 2550 × 2850 | 7,267,500 B |
| res2 | 20m | 1275 × 1425 | 1,816,875 B |
| res5 | 50m | 510 × 570 | 290,700 B |
| res10 | 100m | 255 × 285 | 72,675 B |

## File Naming

```
nuremberg_pred_{YEAR}_res{N}.bin    # predictions
nuremberg_labels_{YEAR}_res{N}.bin  # ground truth labels
```

All files go in: `src/dashboard/data/nuremberg_dashboard/`

## How to Generate a Prediction File (Python)

```python
import numpy as np

# Your model outputs one class index (0-5) per pixel
# Shape: (height, width) at the target resolution
predictions = np.zeros((2850, 2550), dtype=np.uint8)  # res1 example

# Pixels outside Nuremberg boundary → 255
predictions[outside_mask] = 255

# Write raw bytes
predictions.tofile("nuremberg_pred_2026_res1.bin")
```

### Generating All 10 Resolutions

```python
from scipy.stats import mode

# Start from res1 (10m, full pixel resolution)
pred_res1 = ...  # shape (2850, 2550), dtype uint8

for res in range(1, 11):
    if res == 1:
        out = pred_res1
    else:
        # Aggregate by taking the dominant class in each res×res block
        h, w = 2850 // res, 2550 // res
        out = np.full((h, w), 255, dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                block = pred_res1[r*res:(r+1)*res, c*res:(c+1)*res]
                valid = block[block != 255]
                if len(valid) > 0:
                    out[r, c] = mode(valid, keepdims=False).mode
    
    out.tofile(f"nuremberg_pred_2026_res{res}.bin")
```

## Metadata Update

After generating files, add the year to `nuremberg_dashboard_meta.json`:

```json
{
  "prediction_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
}
```

## Coordinate Reference

- **CRS**: EPSG:32632 (UTM zone 32N)
- **Anchor origin**: (641740.0, 5492260.0) — top-left corner
- **Pixel spacing**: 10m
- **Grid**: 2550 columns × 2850 rows

The pixel at row `r`, column `c` in res1 corresponds to UTM coordinates:
```
easting  = 641740 + c * 10
northing = 5492260 - r * 10
```

## Boundary Mask

To determine which pixels are inside/outside Nuremberg, rasterize the boundary GeoJSON:

```python
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

boundary = gpd.read_file("nuremberg_stat_bezirke_wgs84.geojson").to_crs("EPSG:32632")
transform = rasterio.transform.from_origin(641740, 5492260, 10, 10)
mask = rasterize(boundary.geometry, out_shape=(2850, 2550), transform=transform)
outside_mask = (mask == 0)
```
