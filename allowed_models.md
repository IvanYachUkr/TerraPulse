# ✅ Allowed Models & Approaches

> Based on the assignment: *"All models must operate on tabular or fixed-length feature vectors."*

---

## Linear Models
| Model | Library | Use Case |
|-------|---------|----------|
| Ridge Regression | `scikit-learn` | Predict land-cover proportions (continuous) |
| Lasso Regression | `scikit-learn` | Sparse feature selection + prediction |
| Elastic Net | `scikit-learn` | Blend of Ridge + Lasso |
| Linear Regression | `scikit-learn` | Simple baseline |
| Logistic Regression | `scikit-learn` | Binary change/no-change classification |
| Multinomial Logistic Regression | `scikit-learn` | Multi-class land-cover classification |

## Tree-Based / Ensemble Models
| Model | Library | Use Case |
|-------|---------|----------|
| Decision Tree | `scikit-learn` | Interpretable single-tree baseline |
| Random Forest | `scikit-learn` | Ensemble of trees, robust nonlinear model |
| Gradient Boosted Trees (XGBoost) | `xgboost` | High-performance gradient boosting |
| Gradient Boosted Trees (LightGBM) | `lightgbm` | Fast, memory-efficient gradient boosting |
| Gradient Boosted Trees (CatBoost) | `catboost` | Best out-of-box, built-in uncertainty |
| Extra Trees | `scikit-learn` | Random Forest variant |
| AdaBoost | `scikit-learn` | Boosting with weak learners |
| Histogram-Based GB | `scikit-learn` (`HistGradientBoosting*`) | sklearn's built-in fast GB |

## Neural Networks (Tabular Only)
| Model | Library | Use Case |
|-------|---------|----------|
| MLP (Multi-Layer Perceptron) | `scikit-learn` (`MLPRegressor`/`MLPClassifier`) | Simple feedforward NN on tabular features |
| MLP (custom) | `PyTorch`, `JAX/Flax`, `TensorFlow/Keras` | Custom MLP architecture on tabular features |
| TabNet | `pytorch-tabnet` | Attention-based tabular model (still tabular input) |

> [!NOTE]
> Custom MLPs are allowed as long as they operate on **pre-engineered tabular features**, not raw image pixels.

## Simple Temporal Models (Over Tabular Data)
| Model | Library | Use Case |
|-------|---------|----------|
| Autoregressive features + any above model | any | Use T₁ features to predict T₂ labels |
| Temporal difference features | any | Use Δ features as input |
| Rolling/lagged feature engineering | `pandas` | Create temporal context features |

## Other Allowed Approaches
| Approach | Library | Use Case |
|----------|---------|----------|
| SVR / SVC (Support Vector Machines) | `scikit-learn` | Regression/classification on tabular |
| k-Nearest Neighbors | `scikit-learn` | Non-parametric baseline |
| Gaussian Process Regression | `scikit-learn` | Predictions with built-in uncertainty |
| Stacking / Blending Ensembles | `scikit-learn` (`StackingRegressor`) | Combine multiple models |
| Bayesian Ridge Regression | `scikit-learn` | Linear model with uncertainty |

---

## Summary: The Winning Stack

```
🥇 Interpretable:  Ridge Regression (scikit-learn)
🥇 Performance:    CatBoost (catboost)
🏆 Bonus:          JAX/Flax MLP (for engineering showcase)
```
