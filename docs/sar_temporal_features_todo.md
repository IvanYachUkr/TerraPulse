# SAR Temporal Feature Engineering — TODO

## Context
Current SAR features are per-season statistics (mean, std, q25, median, q75, finite_frac) for VV, VH, CR, RVI + LBP texture.
**Missing**: cross-season temporal features that encode vegetation phenology.

## Unit Status (Verified)
- VV/VH stored as **linearly-scaled amplitude** [0,1] (raw DN / 2000, NOT dB)
- CR = VH / VV (linear ratio) — ✅ correct
- RVI = 4·VH/(VV+VH) — ✅ correct
- Misleading docstring says "dB" but code does linear scaling (line 668-678 of `run_multi_city_pipeline_v5.py`)

## Features to Add (in Rust extractor)

### Cross-season amplitude (per cell, computed after all seasons extracted)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| VH_amplitude | max(VH_mean across seasons) - min(...) | Grass fluctuates seasonally, forest stable |
| VV_amplitude | max(VV_mean) - min(VV_mean) | Same for VV |
| CR_amplitude | max(CR_mean) - min(CR_mean) | Ratio stability indicator |
| RVI_amplitude | max(RVI_mean) - min(RVI_mean) | Vegetation index seasonality |

### Summer-winter contrasts
| Feature | Formula | Rationale |
|---------|---------|-----------|
| VH_summer_winter | VH_mean_summer - VH_mean_winter | Direct phenology signal |
| VV_summer_winter | VV_mean_summer - VV_mean_winter | Same |
| CR_summer_winter | CR_mean_summer - CR_mean_winter | Ratio phenology |

### Cross-season variability
| Feature | Formula | Rationale |
|---------|---------|-----------|
| VH_temporal_std | std(VH_mean across all seasons) | Overall temporal variability |
| VV_temporal_std | std(VV_mean across all seasons) | Same |
| CR_temporal_std | std(CR_mean across all seasons) | Same |
| VH_temporal_cv | temporal_std / temporal_mean | Coefficient of variation |
| VV_temporal_cv | temporal_std / temporal_mean | Same |

### Optional (if compute budget allows)
- SAR texture (LBP) temporal variability
- Per-season dB-converted features: `10*log10(linear + eps)` for Gaussian-friendly distributions
- Ascending vs descending orbit separation (requires download code changes)

## Implementation Plan
1. Add `extract_sar_temporal_features()` in `sar_features.rs`
2. Call it from `extract.rs` after all seasons are processed
3. Append ~12-18 new columns to parquet output
4. Re-extract features for all SAR-equipped cities (~15 min)
5. Re-run v5.7 sweep with expanded feature set

## When to Do This
After all 50 cities have SAR data downloaded and ready for extraction.
