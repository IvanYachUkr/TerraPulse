# SpectralSpatialNet V5 notes

## Main idea
Make the model **less obedient to noisy labels** while making the **raw spatial-temporal branch impossible to ignore**.

## What changed

### Architecture
- Replaced global average pooling over the final 3x3 feature map with an explicit **center-context readout**:
  - center feature
  - neighbour mean
  - center minus neighbour mean
- Kept the temporal transformer, but switched to a **CLS token** readout.
- Added **branch dropout** so the handcrafted index branch cannot dominate training.
- Added a **gated fusion** layer.
- Added **auxiliary heads** on the spatial and index branches.

### Training
- Fit **fixed scalers once** on a representative train subset and reuse them for every round.
- Use an **EMA teacher**.
- Use **confidence-aware bootstrapped soft targets**:
  - start with ordinary label-smoothed supervision
  - once the EMA becomes somewhat trustworthy, only soften labels on samples that look suspicious
- **Cap per-sample loss** to stop likely mislabeled pixels from dominating gradients.
- Use **D4 augmentation** (rotations/flips) for the 3x3 patch.
- Select checkpoints by the harmonic mean of:
  - overall accuracy
  - balanced accuracy

## Why this should help
- Your target is the **center pixel**, so the readout now explicitly knows what the center is.
- ESA-style hard labels are noisy, so the loss now treats some labels as suspect instead of absolute truth.
- The model can no longer solve the task almost entirely through the center-pixel index branch.
- Checkpoint selection no longer rewards models that get easy dominant classes right while butchering the minority ones.

## Files
- `spectral_spatial_v5.py`
- `train_ssnet_v5.py`

## Integration
Place `spectral_spatial_v5.py` into your architecture module path and update imports accordingly.
