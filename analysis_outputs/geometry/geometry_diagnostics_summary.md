# Geometry Diagnostics for Country

## Why not a ring?

- In the PC1/PC2 residual plot, positives occupy a slanted band rather than wrapping around an empty center.
- Radius in PC1/PC2 is broad: mean=0.776, std=0.436, coefficient_of_variation=0.562.
- Angle coverage is visibly uneven in `ring_diagnostics_radius_angle.png`.

## Why not clearly disconnected clusters?

- The PCA views show a mostly continuous positive cloud/band, not separated islands with empty gaps.
- Quick clustering diagnostics on positive residual PCs are weak rather than decisive:
  - kmeans_k=2 silhouette=0.4518
  - kmeans_k=3 silhouette=0.3867
  - kmeans_k=4 silhouette=0.3328
  - kmeans_k=5 silhouette=0.3418
  - kmeans_k=6 silhouette=0.3133
  - dbscan_eps=0.10 clusters=3 noise_frac=0.962
  - dbscan_eps=0.15 clusters=8 noise_frac=0.421
  - dbscan_eps=0.20 clusters=1 noise_frac=0.091
  - dbscan_eps=0.25 clusters=1 noise_frac=0.025
  - dbscan_eps=0.30 clusters=1 noise_frac=0.009
  - dbscan_eps=0.40 clusters=1 noise_frac=0.003

## Why low-dimensional manifold-like band?

- Positive-only residual PCA first six EVR: [0.8228, 0.0979, 0.0433, 0.0217, 0.0086, 0.0057]
- Positive-only cumulative EVR first six: [0.8228, 0.9207, 0.964, 0.9857, 0.9943, 1.0]
- PC1+PC2 explain more than 90% of positive residual variance, so the positives are concentrated near a 2D structure.
- The positive points have a continuous slanted trend in residual PC1/PC2, illustrated with the quadratic trend plot.
- Quadratic trend coefficients for PC2=f(PC1): [-0.005, 0.1907, 0.0013]
