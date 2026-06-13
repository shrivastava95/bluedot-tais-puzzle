# Visualizing the 2D Spline Shape for `country`

These plots fit 2D spline-like charts to the `country` positive residual activations, decode grids from each chart back into the 64D residual activation space, then project the decoded points into global residual PCA coordinates for visualization.

## Single-Chart View

A single 2D chart gives a rough global sheet/band through the positives:

![single chart 2D](single_2d_spline_sheet_2d.png)

![single chart 3D](single_2d_spline_sheet_3d.png)

## Multi-Chart View

A union of six 2D charts captures local variation better. The charts tile a continuous band-like region rather than forming a clean circle or obviously disconnected islands:

![union charts 2D](union_2d_spline_charts_2d.png)

![union charts 3D](union_2d_spline_charts_3d.png)

## Local Latent Domains

Each chart has its own 2D latent PCA domain. Decoding these domains back into residual space gives the colored chart patches above:

![local domains](local_2d_chart_domains.png)

## Interpretation

This is only a projection of a 64D object, not a literal surface in 3D. Still, it shows the practical shape implied by the K=2 spline model: a low-dimensional band/sheet running through the country-positive region. The six-chart version is the better visual model because it allows local patches to bend and vary while staying effectively 2D.
