# ❌ Not Allowed Models & Approaches

> Based on the assignment: *"End-to-end neural network models (e.g. CNNs/Transformers) are not allowed."*
> *"All models must operate on tabular or fixed-length feature vectors."*
> *"Features must be engineered or aggregated from imagery and auxiliary data."*

---

## Explicitly Banned

### Convolutional Neural Networks (CNNs)
| Model | Why Banned |
|-------|-----------|
| ResNet, VGG, EfficientNet | Operate directly on raw image pixels |
| U-Net, SegNet, DeepLab | Semantic segmentation on raw imagery |
| Any CNN-based architecture | End-to-end image → label is not allowed |

### Transformers & Attention Models (on images)
| Model | Why Banned |
|-------|-----------|
| Vision Transformer (ViT) | Operates on image patches directly |
| Swin Transformer | Hierarchical vision transformer |
| Segment Anything (SAM) | Foundation model for image segmentation |
| Any image-based Transformer | End-to-end image processing |

### Pre-trained Foundation Models (on images)
| Model | Why Banned |
|-------|-----------|
| CLIP (for image features) | Uses CNN/Transformer backbone on raw images |
| SatMAE, Prithvi | Satellite-specific foundation models on raw imagery |
| Any model that takes raw pixels as input | Defeats the tabular feature engineering requirement |

---

## Grey Area — Technically Not Allowed

| Approach | Why It's Problematic |
|----------|---------------------|
| Using a pre-trained CNN to **extract features** then feeding into a tabular model | The CNN part is doing the feature engineering — the assignment requires **you** to engineer features |
| Autoencoders on raw imagery | Still an end-to-end neural approach on images |
| Graph Neural Networks on pixel adjacency | Operates on raw spatial structure, not tabular features |
| RNNs / LSTMs on pixel sequences | Sequence model on raw data, not on tabular features |

> [!WARNING]
> The key rule is: **you must engineer the features yourself** from the imagery (spectral indices, zonal statistics, etc.), then feed those tabular features to an allowed model. Any model that bypasses this step by learning directly from raw image data is not allowed.

---

## The Line in the Sand

```
Raw satellite image → [YOUR feature engineering] → Tabular features → Allowed model ✅
Raw satellite image → CNN/Transformer → Prediction                                  ❌
Raw satellite image → Pre-trained CNN → Embeddings → Tabular model                   ❌ (grey area, avoid)
```
