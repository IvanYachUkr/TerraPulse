"""
Grid Size Analysis: Find optimal spatial unit for the tabular ML dataset.

For each candidate grid size, we compute:
  1. Number of grid cells (samples)
  2. Within-cell class entropy (lower = more homogeneous cells = cleaner signal)
  3. Between-cell class variance (higher = more diverse cells = better for ML)
  4. Fraction of "pure" cells (>80% one class) vs. "mixed" cells
  5. Class proportion variance (do we have enough variation to learn from?)
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Load WorldCover 2020 clipped to Nuremberg
BBOX = [10.95, 49.38, 11.20, 49.52]  # west, south, east, north
ds = rasterio.open("data/labels/ESA_WorldCover_2020_N48E009.tif")
window = from_bounds(*BBOX, ds.transform)
data = ds.read(1, window=window)
transform = ds.window_transform(window)
print(f"WorldCover tile clipped: {data.shape} at {abs(transform.a):.4f}° per pixel")

# Pixel size in meters (approx at 49°N latitude)
# 10m resolution
PIXEL_SIZE = 10  # meters

# Our 6 merged classes
CLASS_MAP = {
    10: 0,   # Tree cover -> 0
    30: 1,   # Grassland -> 1
    90: 1,   # Wetland -> Grassland
    40: 2,   # Cropland -> 2
    50: 3,   # Built-up -> 3
    60: 4,   # Bare/sparse -> 4
    80: 5,   # Water -> 5
}
CLASS_NAMES = ["Tree cover", "Grassland", "Cropland", "Built-up", "Bare/sparse", "Water"]
N_CLASSES = 6

# Remap pixels to our 6 classes
remapped = np.full_like(data, fill_value=255, dtype=np.uint8)
for src, dst in CLASS_MAP.items():
    remapped[data == src] = dst

print(f"Remapped pixels: {np.sum(remapped < N_CLASSES)} / {data.size} valid")

# Candidate grid sizes
GRID_SIZES_M = [50, 100, 150, 200, 300, 500]

print(f"\n{'='*90}")
print(f"{'Grid (m)':>10} {'Cells':>8} {'Avg Entropy':>13} {'Frac Pure>80%':>14} {'Frac Pure>90%':>14} {'Avg Max%':>10}")
print(f"{'='*90}")

for grid_m in GRID_SIZES_M:
    grid_px = grid_m // PIXEL_SIZE  # pixels per grid cell side
    
    n_rows = remapped.shape[0] // grid_px
    n_cols = remapped.shape[1] // grid_px
    n_cells = n_rows * n_cols
    
    entropies = []
    max_fracs = []
    pure80 = 0
    pure90 = 0
    class_proportions = []  # per-cell class proportions
    
    for i in range(n_rows):
        for j in range(n_cols):
            cell = remapped[i*grid_px:(i+1)*grid_px, j*grid_px:(j+1)*grid_px]
            valid = cell[cell < N_CLASSES]
            
            if len(valid) == 0:
                continue
            
            # Class proportions for this cell
            props = np.zeros(N_CLASSES)
            for c in range(N_CLASSES):
                props[c] = np.sum(valid == c) / len(valid)
            
            class_proportions.append(props)
            
            # Entropy (measure of mixing)
            props_nonzero = props[props > 0]
            entropy = -np.sum(props_nonzero * np.log2(props_nonzero))
            entropies.append(entropy)
            
            # Max fraction (how "pure" is this cell?)
            max_frac = props.max()
            max_fracs.append(max_frac)
            if max_frac > 0.8:
                pure80 += 1
            if max_frac > 0.9:
                pure90 += 1
    
    class_proportions = np.array(class_proportions)
    avg_entropy = np.mean(entropies)
    avg_max_frac = np.mean(max_fracs)
    
    print(f"{grid_m:>8}m {n_cells:>8,} {avg_entropy:>13.3f} {pure80/n_cells:>13.1%} {pure90/n_cells:>13.1%} {avg_max_frac:>9.1%}")

# Detailed analysis for top candidates
print(f"\n{'='*90}")
print("DETAILED ANALYSIS: Class proportion variance (higher = more variation to learn from)")
print(f"{'='*90}")

for grid_m in [100, 200, 300]:
    grid_px = grid_m // PIXEL_SIZE
    n_rows = remapped.shape[0] // grid_px
    n_cols = remapped.shape[1] // grid_px
    
    class_proportions = []
    for i in range(n_rows):
        for j in range(n_cols):
            cell = remapped[i*grid_px:(i+1)*grid_px, j*grid_px:(j+1)*grid_px]
            valid = cell[cell < N_CLASSES]
            if len(valid) == 0:
                continue
            props = np.zeros(N_CLASSES)
            for c in range(N_CLASSES):
                props[c] = np.sum(valid == c) / len(valid)
            class_proportions.append(props)
    
    class_proportions = np.array(class_proportions)
    
    print(f"\n  Grid: {grid_m}m × {grid_m}m  ({len(class_proportions)} cells)")
    print(f"  {'Class':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'IQR':>8}")
    for c in range(N_CLASSES):
        col = class_proportions[:, c]
        q25, q75 = np.percentile(col, [25, 75])
        print(f"  {CLASS_NAMES[c]:<15} {col.mean():>7.1%} {col.std():>7.1%} {col.min():>7.1%} {col.max():>7.1%} {q75-q25:>7.1%}")
    
    # How many cells have "meaningful" change signal?
    # (i.e., cells where built-up is between 10-90% — mixed urban/non-urban)
    built_col = class_proportions[:, 3]
    mixed = np.sum((built_col > 0.1) & (built_col < 0.9))
    print(f"  Mixed urban cells (10-90% built-up): {mixed} ({mixed/len(class_proportions):.1%})")

print(f"\n{'='*90}")
print("RECOMMENDATION")
print(f"{'='*90}")
print("Look for grid size that maximizes:")
print("  - Enough cells for ML (>500 ideally)")  
print("  - Good mix of pure + mixed cells (not all pure, not all mixed)")
print("  - High class proportion variance (Std) for learning signal")
print("  - Enough 'mixed urban' cells to model urbanization transitions")
