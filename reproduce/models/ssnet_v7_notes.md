# SSNet V7 notes

## What V7 is trying to do

V7 is **not** another V5 student.

It keeps the good V5 engineering:
- fixed scalers
- fast / stable iterative resampling
- center-aware 3x3 readout
- confidence-aware diagnostics

But changes the actual modeling assumption:
- labels are not trusted equally
- ambiguous boundary pixels should not dominate the loss
- explicit spectral logic should be allowed to push back on ESA

## Core additions

### 1. Raw spectral prior head
V7 takes the **raw unscaled center pixel** (`center_raw`) and computes compact temporal summaries of:
- NDVI
- NDWI
- MNDWI
- NDBI
- BSI
- NDMI
- NBR2

Then a learnable soft-threshold bank builds fuzzy rules like:
- "above threshold"
- "below threshold"

This becomes `prior_logits`.

### 2. High-precision heuristic anchors
V7 also creates **conservative anchor logits** for the 7 classes from those indices.
These are only meant to be strong on obvious pixels.
They are used to:
- soften labels when ESA looks suspicious
- supervise the prior branch
- bias final logits on very clear cases

### 3. Ambiguity head
The model predicts an ambiguity score from:
- spatial-temporal features
- index features
- prior features
- patch heterogeneity

Hard labels are downweighted more on ambiguous / boundary-like pixels.

## Important implementation detail

V7 needs one extra input beyond V5:
- `center_raw`: the **raw unscaled** 6×12 center-pixel bands

Do **not** compute the prior head from scaled patch values. That would make the physical indices meaningless.

## Default band-order assumption

The prior head uses our actual 12-band order per timestep:

`[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VV, VH]`

(10 Sentinel-2 spectral + 2 Sentinel-1 SAR — **no B01/B09**)

That means internally it uses:
- blue = 0 (B02)
- green = 1 (B03)
- red = 2 (B04)
- nir = 6 (B08)
- swir1 = 8 (B11)
- swir2 = 9 (B12)

`DEFAULT_BAND_MAP` in `spectral_spatial_v7.py` has been corrected to match this.

## Suggested first run

```bash
python 05_train_ssnet_v7.py
```

If you want checkpointing to care more about shrubland / bare_sparse:

```bash
python 05_train_ssnet_v7.py --ckpt-metric focus_score
```

## What I would watch during training

- `alpha`: mean anchor-mix strength into the target
- `alpha_nz`: fraction of samples getting nonzero anchor mixing
- `anc`: mean anchor confidence
- `anc_frac`: fraction of samples above anchor threshold
- `relbl`: fraction of samples where anchor strongly disagrees with ESA
- `bnd`: average boundary / heterogeneity score
- `amb`: average learned ambiguity
- `pg`: average prior-gate value
- `hard`: mean recall of shrubland + bare_sparse
- `prio`: mean recall of shrubland + bare_sparse + built_up + water

## Practical expectation

V7 is meant to push **farther from pure ESA obedience** than V5/V6.

That does **not** guarantee a better official metric on noisy labels.
What it tries to do is:
- preserve or improve actual map quality
- stop obviously bad label pressure on mixed pixels
- let explicit spectral logic correct some water / built-up / bare / vegetation mistakes
