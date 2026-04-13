# SpectralSpatialNet V8

## What changed from V7.1

V7.1 improved coherence and boundaries, but it sometimes over-smoothed small true islands of another class.
Examples the user specifically called out:
- tiny or narrow **water** pixels inside mostly grass / urban context
- small local exceptions that the neighborhood branch wanted to absorb into the surrounding class

V8 is designed to keep the good parts of V7.1 while adding an explicit **center-only veto**.

### Core idea
Use three complementary signals:
1. **Neighborhood-aware neural branch**
   - good for coherent regions and boundary geometry
2. **Spectral prior branch**
   - soft threshold logic over physically meaningful EO indices
3. **Center specialist branch**
   - a pixel-level expert that focuses on the center cell only
   - meant to rescue real local exceptions from being smoothed away

The final logits are:

```text
final = neural_logits
      + prior_gate * prior_logits
      + anchor_scale * anchor_strength * anchor_anchor_logits
      + center_scale * center_gate * center_logits
```

## Why this is different from V7.1

V7.1 mostly had:
- prior-guided softening
- ambiguity-aware downweighting
- a tendency to prefer coherent regions

V8 adds an explicit mechanism for:
- "the neighborhood says class A"
- but "the center pixel itself really looks like class B"

That is the purpose of the center expert + anomaly-aware center gate.

## New components

### 1. Center specialist expert
Inputs:
- scaled center pixel time-series (`72 = 6 timesteps * 12 bands`)
- scaled engineered indices (`145`)

Outputs:
- `center_branch_logits`
- `center_conf`

This expert is intended to become strong exactly where spatial smoothing is dangerous.

### 2. Boundary vs anomaly split
V7.1 had ambiguity and boundary-style reasoning, but V8 makes a more explicit distinction:
- **boundary_score**: mixed / boundary-like region where labels should be trusted less
- **anomaly_score**: center pixel differs from neighbors enough that it may be a real local exception

This is crucial because the training treatment should differ:
- boundary pixel -> softer supervision / less confidence
- anomaly pixel -> *do not* suppress it into the neighborhood too aggressively

### 3. Center gate
`center_gate` is learned and further modulated by:
- center confidence
- anomaly score

So the center expert can vote strongly only when:
- it is confident
- and the pixel actually looks like a local exception

## Training objective

V8 still does not treat ESA as holy truth.
It uses:
- soft cross-entropy to mixed targets
- prior-guided target blending
- center-guided target blending on anomaly pixels
- ambiguity-aware sample weighting
- explicit auxiliary losses for:
  - spatial branch
  - index branch
  - prior branch
  - center branch
  - ambiguity/gates

### Important difference from V7.1
Ambiguous boundary pixels are downweighted, but anomaly pixels receive a partial **upweight** so that real local exceptions are not erased by spatial coherence.

## Suggested first run

Train from scratch (default):

```bash
python 05_train_ssnet_v8.py --n-rounds 8 --max-epochs 20 --batch-size 2048 --ckpt-metric score
```

If you later want to push more on hard classes specifically:

```bash
python 05_train_ssnet_v8.py --n-rounds 8 --max-epochs 20 --batch-size 2048 --ckpt-metric focus_score
```

## What to watch in logs

Useful diagnostics:
- `anc_frac` and `alpha_nz`: prior anchor coverage
- `ctr_frac` and `ctr_conf`: how often the center expert is confident
- `anom`: anomaly score mean
- `cg`: center gate mean
- `pg`: prior gate mean

Healthy behavior should look like:
- center gate not collapsed to 0
- anomaly score nontrivial
- center branch active on some subset, not everywhere
- no runtime cliff / shared GPU memory spill

## Practical expectation

V8 is not a pure "more aggressive priors" model.
It is specifically trying to fix:
- over-smoothing of small true local exceptions
- while preserving V7.1's better region coherence and river boundaries

So if it works, the likely visual improvement is:
- better tiny water and local exception recovery
- without going back to noisy square-by-square fragmentation
