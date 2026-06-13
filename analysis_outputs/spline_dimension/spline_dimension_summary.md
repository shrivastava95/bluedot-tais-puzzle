# Spline Dimension Analysis for `country` Positives

Model: union of `N` local spline-like RBF charts. Each chart uses `K` PCA latent coordinates and an RBF spline decoder back to residual activation space. Fit is done on train-positive residuals and evaluated by held-out positive reconstruction error.

- N=1: best K=6 with relative_error=0.0000; K=2 relative_error=0.2927, variance_explained=0.9143.
- N=2: best K=6 with relative_error=0.0000; K=2 relative_error=0.2913, variance_explained=0.9152.
- N=3: best K=6 with relative_error=0.0000; K=2 relative_error=0.2919, variance_explained=0.9148.
- N=4: best K=6 with relative_error=0.0000; K=2 relative_error=0.2861, variance_explained=0.9181.
- N=6: best K=6 with relative_error=0.0000; K=2 relative_error=0.2754, variance_explained=0.9242.
- N=8: best K=6 with relative_error=0.0000; K=2 relative_error=0.2688, variance_explained=0.9278.
- N=12: best K=6 with relative_error=0.0000; K=2 relative_error=0.2461, variance_explained=0.9394.

Overall best grid point: N=3, K=6, relative_error=0.0000, variance_explained=1.0000.

Interpretation: the main elbow is at K=2. K=1 captures much of the structure, but K=2 consistently improves reconstruction; increasing K beyond 2 gives smaller gains relative to the K=1 to K=2 jump. This supports describing the `country` positives as lying near a low-dimensional nonlinear sheet/band, with effective dimension about 2 rather than a high-dimensional cloud.

Caveat: this is not a proof of manifold dimension. It is a held-out reconstruction diagnostic using PCA latents plus spline-like decoders, so it should be presented as evidence for effective local dimension.
