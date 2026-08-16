## Overview
KNN is a fundamentally different kind of algorithm from the previous two — it's **instance-based (lazy) learning**: there's no explicit training phase that fits parameters. Instead, the entire training dataset is stored, and predictions are made at inference time by finding the K most similar (nearest) training points to a new observation and aggregating their labels. The core assumption is simple and intuitive: similar things exist close together in feature space.

## How It Works
1. Choose a value for **K** (the number of neighbors to consider) and a **distance metric** (commonly Euclidean distance)
2. For a new observation, compute its distance to every point in the training set
3. Select the K closest points
4. **Classification:** predict the majority class among those K neighbors (often weighted by inverse distance, so closer neighbors count more)
5. **Regression:** predict the average (or distance-weighted average) of the K neighbors' target values

## Methods & Techniques
- **Choosing K:** the single most important hyperparameter.
  - Small K (e.g. K=1) → low bias, high variance — very sensitive to noise, decision boundary is jagged, prone to overfitting
  - Large K → high bias, low variance — smoother decision boundary, but risks oversimplifying and blending together genuinely different regions
  - Standard practice: use **cross-validation** to sweep a range of K values and pick the one minimizing validation error; using an **odd K** for binary classification avoids tie votes
- **Feature scaling is mandatory, not optional:** since KNN relies directly on distance calculations, unscaled features with larger numeric ranges will dominate the distance metric regardless of their actual importance. **StandardScaler** or **MinMaxScaler** must be applied before fitting.
- **Distance metric choice:**
  - **Euclidean distance** — the default, works well for continuous features
  - **Manhattan distance** — sometimes more robust in high-dimensional space
  - **Minkowski distance** — a generalization that includes both as special cases
  - **Hamming distance** — for categorical/binary features
- **The curse of dimensionality:** KNN's core assumption (nearby points are similar) breaks down as the number of features grows — in high dimensions, distances between points tend to converge, making "nearest" neighbors not meaningfully closer than "far" ones. Dimensionality reduction (PCA) or careful feature selection becomes important as feature count grows.
- **Weighted voting:** instead of a simple majority vote, weight each neighbor's vote by the inverse of its distance (`weights='distance'` in scikit-learn) so closer neighbors have proportionally more influence — often improves performance over uniform voting
- **Efficient neighbor search:** for large datasets, brute-force distance computation to every point is slow (O(n) per query) — **KD-Trees** or **Ball Trees** speed up nearest-neighbor lookup for lower-dimensional data (their advantage shrinks in very high dimensions)

## Evaluation Metrics
Same as standard classification (Accuracy, Precision, Recall, F1, ROC-AUC) or regression (RMSE, MAE, R²) metrics, depending on the task — plus a K-vs-validation-error curve to visualize the bias-variance trade-off across K values.

## When to Use It
Good for smaller-to-medium datasets with a clear notion of feature-space similarity, as a simple, non-parametric baseline that makes no assumptions about the underlying data distribution, and in domains like recommendation systems or anomaly detection where "similarity to known examples" is a natural framing.

## Strengths & Limitations
| Strengths | Limitations |
|---|---|
| Simple, intuitive, no training phase | Slow at prediction time on large datasets (must compare to every training point) |
| Makes no assumptions about data distribution | Requires feature scaling |
| Naturally handles multi-class problems | Suffers badly from the curse of dimensionality |
| Can capture complex, nonlinear decision boundaries | Sensitive to irrelevant/noisy features and class imbalance |

---
---
