# Rust COG Reader & Download: Known Issues

Issues discovered during multi-year Nuremberg timeseries generation (Feb 2025).
Current workaround: Python (stackstac/rasterio) downloads, Rust extract+predict.

---

## Issue 1: Post-2022 S2 Pixel Value Inflation (5-26× scaling)

**Status**: OPEN — workaround in place (Python download)

**Symptom**: Sentinel-2 composites from Planetary Computer for years ≥ 2022
produce pixel values 5-26× higher than expected when downloaded by the Rust
COG reader. For example, B02 median for 2022 spring = 4882 (expected ~473+1000
= ~1473 pre-offset-correction, or ~473 post-correction).

**Impact**: Predictions for 2022 were wildly incorrect (built_up 61% instead of
~30%, water 0.1% instead of ~1%). Python stackstac produces correct values.

**Root cause**: Not fully identified. The Rust COG reader (`cog.rs`) parses
TIFF IFDs and reads tiles correctly for pre-2022 data. For post-2022 data,
something changes in the COG tile structure or overview levels:

- Same resolution (10m), same dimensions (10980×10980), same CRS (EPSG:32632)
- Raw rasterio reads show 2022 B02 p50 ~1238 (correct), but Rust composite
  produces p50 ~4882 (inflated)
- 2024 data (15bps COG format) shows different inflation factor (0.86× or 26×
  depending on season)

**Possible causes to investigate**:
1. Overview level selection — newer COGs may have different IFD chain structure
2. Tile byte offset parsing — BigTIFF vs classic TIFF differences
3. Pixel value interpretation with 15bps samples
4. Compression handling differences (DEFLATE parameters)
5. nodata value handling — newer COGs use nodata=0.0 vs None

**Fix path**: Compare byte-level tile data between rasterio and Rust reader
for a single tile from a 2022 scene. Check if the IFD being parsed points
to the correct overview level.

---

## Issue 2: 15bps (15-bit) COG Format

**Status**: PARTIALLY FIXED — treated as uint16

**Symptom**: Newer S2 COGs from Planetary Computer have `BitsPerSample: 15`
(instead of 16). The Rust reader originally didn't support this.

**Fix applied**: Added `(15, 1)` case to `decode_tile()` in `cog.rs`, treating
15bps as uint16. This allows reading the data, but may contribute to the
scaling issue (Issue 1) if the 15-bit packing differs from standard uint16.

**Remaining concern**: Verify that 15bps tiles are packed identically to 16bps
in the byte layout. The TIFF spec says BitsPerSample describes the number of
significant bits, but storage is still in uint16 containers. However, some
encoders may pack 15-bit values differently.

**Location**: `terrapulse/src/cog.rs`, `decode_tile()` function.

---

## Issue 3: SAR LERC Compression (code 50000) Not Supported

**Status**: OPEN — workaround in place (Python download)

**Symptom**: Sentinel-1 SAR scenes from 2024-2025 (and partially 2023) use
LERC compression (`tiff_compression: 50000`), which the Rust COG reader
doesn't support. All SAR downloads for these years fail with
"Unsupported TIFF compression: 50000".

**Impact**: No SAR data for 2024-2025, preventing prediction for year pairs
involving these years (model requires SAR features).

**Fix path**: Two options:
1. **Add LERC decompression** via the `lerc` crate or FFI to the C LERC
   library. LERC is an open standard by Esri.
2. **Use Python for SAR download** (current workaround) — rasterio uses
   GDAL which supports LERC natively.

**Location**: `terrapulse/src/cog.rs`, `decode_tile()` match on compression.

---

## Issue 4: ESA Processing Baseline BOA_ADD_OFFSET

**Status**: FIXED (per-scene detection) — but superseded by Issue 1

**Symptom**: Starting Jan 25, 2022 (ESA Processing Baseline 04.00), S2 L2A
products include a +1000 DN offset on all spectral bands. Composites mixing
old and new baseline scenes produce inconsistent pixel values.

**Fix applied**: Per-scene auto-detection in `composite.rs`. Checks B02
median; if > 900, subtracts 1000 from all spectral bands before compositing.
This correctly handles mixed-baseline years like 2022.

**Note**: This fix works correctly in isolation, but is insufficient because
Issue 1 (the scaling bug) corrupts the values before the offset check runs.
When Issue 1 is resolved, this fix should work as intended.

**Location**: `terrapulse/src/composite.rs`, `download_and_composite()`.

---

## Issue 5: SAR Border NaN Values (42K+ imputed)

**Status**: OPEN — cosmetic, doesn't break predictions

**Symptom**: Year pairs involving 2021-2022 SAR show 42K NaN values during
feature extraction. The Python pipeline doesn't produce this many NaNs.

**Root cause**: Likely the Rust SAR reprojection (bilinear resampling from
GCP-referenced scenes) handles tile edges differently than rasterio's
WarpedVRT, producing NaN borders on some scenes.

**Impact**: Low — NaN values are imputed during extraction. Predictions
still work, but edge cells may have lower-quality features.

**Location**: `terrapulse/src/download.rs`, SAR download/reproject logic.

---

## Summary Table

| Issue | Severity | Status | Workaround |
|-------|----------|--------|------------|
| 1. S2 pixel inflation (2022+) | **Critical** | OPEN | Python download |
| 2. 15bps COG format | Medium | Partial fix | Treated as uint16 |
| 3. SAR LERC compression | **High** | OPEN | Python download |
| 4. BOA_ADD_OFFSET | Medium | Fixed | Per-scene detection |
| 5. SAR border NaN | Low | OPEN | Imputation |
