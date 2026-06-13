import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_argmin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path("analysis_outputs")
SHAPE_OUT = OUT / "spline_shape"
SHAPE_OUT.mkdir(exist_ok=True)


def load_cached():
    names = json.load(open("feature_names.json"))
    tr = np.load(OUT / "train_acts.npz")
    te = np.load(OUT / "test_acts.npz")
    return names, tr["acts"], tr["labels"], te["acts"], te["labels"]


def residualize(names, Xtr, Ytr, Xte, suspect="country"):
    coefs = []
    for i, name in enumerate(names):
        if name == suspect:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=0),
        )
        clf.fit(Xtr, Ytr[:, i])
        scaler = clf.named_steps["standardscaler"]
        lr = clf.named_steps["logisticregression"]
        coefs.append(lr.coef_[0] / scaler.scale_)
    W = np.stack(coefs, axis=1)
    Q, _ = np.linalg.qr(W)
    return Xtr - Xtr @ Q @ Q.T, Xte - Xte @ Q @ Q.T


def grid_inside_hull(U, grid_n=35, margin=0.03):
    mins = U.min(axis=0)
    maxs = U.max(axis=0)
    span = maxs - mins
    xs = np.linspace(mins[0] - margin * span[0], maxs[0] + margin * span[0], grid_n)
    ys = np.linspace(mins[1] - margin * span[1], maxs[1] + margin * span[1], grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    if len(U) >= 4:
        hull = ConvexHull(U)
        poly = U[hull.vertices]
        mask = MplPath(poly).contains_points(pts)
    else:
        mask = np.ones(len(pts), dtype=bool)
    return pts[mask]


def fit_charts(Xpos, n_charts=6, latent_dim=2):
    km = KMeans(n_clusters=n_charts, n_init=20, random_state=0)
    labels = km.fit_predict(Xpos)
    charts = []
    for c in range(n_charts):
        Xc = Xpos[labels == c]
        pca = PCA(n_components=latent_dim, random_state=0).fit(Xc)
        U = pca.transform(Xc)
        neighbors = min(80, max(latent_dim + 5, len(Xc)))
        rbf = RBFInterpolator(U, Xc, kernel="cubic", smoothing=1e-4, neighbors=neighbors)
        charts.append({"X": Xc, "U": U, "pca": pca, "rbf": rbf, "center": km.cluster_centers_[c]})
    return charts, km


def decode_chart_grids(charts, global_pca, grid_n=38):
    decoded = []
    for c, chart in enumerate(charts):
        Ugrid = grid_inside_hull(chart["U"], grid_n=grid_n)
        Xgrid = chart["rbf"](Ugrid)
        Zgrid = global_pca.transform(Xgrid)
        Zdata = global_pca.transform(chart["X"])
        decoded.append({"chart": c, "Ugrid": Ugrid, "Xgrid": Xgrid, "Zgrid": Zgrid, "Zdata": Zdata})
    return decoded


def plot_single_sheet(Xres_test, y_test, Xpos_train, decoded_single, global_pca):
    Ztest = global_pca.transform(Xres_test)
    Zpos_train = global_pca.transform(Xpos_train)
    sheet = decoded_single[0]["Zgrid"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Ztest[y_test == 0, 0], Ztest[y_test == 0, 1], Ztest[y_test == 0, 2], s=5, alpha=0.06, color="#7aa6c2", label="negative")
    ax.scatter(Zpos_train[:, 0], Zpos_train[:, 1], Zpos_train[:, 2], s=7, alpha=0.18, color="#ff7f0e", label="country positive")
    ax.scatter(sheet[:, 0], sheet[:, 1], sheet[:, 2], s=8, alpha=0.55, color="black", label="decoded 2D spline sheet")
    ax.set_xlabel("residual PC1")
    ax.set_ylabel("residual PC2")
    ax.set_zlabel("residual PC3")
    ax.set_title("Single 2D spline chart decoded into residual PCA space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SHAPE_OUT / "single_2d_spline_sheet_3d.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Ztest[y_test == 0, 0], Ztest[y_test == 0, 1], s=7, alpha=0.12, color="#7aa6c2", label="negative")
    ax.scatter(Zpos_train[:, 0], Zpos_train[:, 1], s=8, alpha=0.25, color="#ff7f0e", label="country positive")
    ax.scatter(sheet[:, 0], sheet[:, 1], s=9, alpha=0.65, color="black", label="decoded 2D spline chart")
    ax.set_xlabel("residual PC1")
    ax.set_ylabel("residual PC2")
    ax.set_title("Single decoded 2D spline chart: PC1/PC2 view")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SHAPE_OUT / "single_2d_spline_sheet_2d.png", dpi=180)
    plt.close(fig)


def plot_union_charts(Xres_test, y_test, Xpos_train, decoded, global_pca):
    Ztest = global_pca.transform(Xres_test)
    Zpos_train = global_pca.transform(Xpos_train)
    colors = plt.cm.tab10(np.linspace(0, 1, len(decoded)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Ztest[y_test == 0, 0], Ztest[y_test == 0, 1], Ztest[y_test == 0, 2], s=5, alpha=0.04, color="#7aa6c2", label="negative")
    ax.scatter(Zpos_train[:, 0], Zpos_train[:, 1], Zpos_train[:, 2], s=6, alpha=0.09, color="#ff7f0e", label="country positive")
    for d, color in zip(decoded, colors):
        rgba = to_rgba(color, alpha=0.68)
        ax.scatter(d["Zgrid"][:, 0], d["Zgrid"][:, 1], d["Zgrid"][:, 2], s=8, color=rgba, label=f"chart {d['chart']}")
    ax.set_xlabel("residual PC1")
    ax.set_ylabel("residual PC2")
    ax.set_zlabel("residual PC3")
    ax.set_title("Union of six 2D spline charts decoded into residual PCA space")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(SHAPE_OUT / "union_2d_spline_charts_3d.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Ztest[y_test == 0, 0], Ztest[y_test == 0, 1], s=7, alpha=0.10, color="#7aa6c2", label="negative")
    ax.scatter(Zpos_train[:, 0], Zpos_train[:, 1], s=8, alpha=0.16, color="#ff7f0e", label="country positive")
    for d, color in zip(decoded, colors):
        ax.scatter(d["Zgrid"][:, 0], d["Zgrid"][:, 1], s=10, alpha=0.70, color=color, label=f"chart {d['chart']}")
    ax.set_xlabel("residual PC1")
    ax.set_ylabel("residual PC2")
    ax.set_title("Union of six decoded 2D spline charts: PC1/PC2 view")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(SHAPE_OUT / "union_2d_spline_charts_2d.png", dpi=180)
    plt.close(fig)


def plot_latent_domains(charts):
    n = len(charts)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.6 * rows))
    axes = np.array(axes).ravel()
    for c, chart in enumerate(charts):
        ax = axes[c]
        U = chart["U"]
        ax.scatter(U[:, 0], U[:, 1], s=9, alpha=0.55)
        if len(U) >= 4:
            hull = ConvexHull(U)
            poly = U[hull.vertices]
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color="black", linewidth=1)
        ax.set_title(f"chart {c} latent 2D domain")
        ax.set_xlabel("local latent u1")
        ax.set_ylabel("local latent u2")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Local 2D spline domains for country-positive residuals")
    fig.tight_layout()
    fig.savefig(SHAPE_OUT / "local_2d_chart_domains.png", dpi=180)
    plt.close(fig)


def write_summary():
    (SHAPE_OUT / "spline_shape_visualization.md").write_text(
        "# Visualizing the 2D Spline Shape for `country`\n\n"
        "These plots fit 2D spline-like charts to the `country` positive residual activations, decode grids from each chart back into the 64D residual activation space, then project the decoded points into global residual PCA coordinates for visualization.\n\n"
        "## Single-Chart View\n\n"
        "A single 2D chart gives a rough global sheet/band through the positives:\n\n"
        "![single chart 2D](single_2d_spline_sheet_2d.png)\n\n"
        "![single chart 3D](single_2d_spline_sheet_3d.png)\n\n"
        "## Multi-Chart View\n\n"
        "A union of six 2D charts captures local variation better. The charts tile a continuous band-like region rather than forming a clean circle or obviously disconnected islands:\n\n"
        "![union charts 2D](union_2d_spline_charts_2d.png)\n\n"
        "![union charts 3D](union_2d_spline_charts_3d.png)\n\n"
        "## Local Latent Domains\n\n"
        "Each chart has its own 2D latent PCA domain. Decoding these domains back into residual space gives the colored chart patches above:\n\n"
        "![local domains](local_2d_chart_domains.png)\n\n"
        "## Interpretation\n\n"
        "This is only a projection of a 64D object, not a literal surface in 3D. Still, it shows the practical shape implied by the K=2 spline model: a low-dimensional band/sheet running through the country-positive region. The six-chart version is the better visual model because it allows local patches to bend and vary while staying effectively 2D.\n"
    )


def main():
    names, Xtr, Ytr, Xte, Yte = load_cached()
    Xtr_res, Xte_res = residualize(names, Xtr, Ytr, Xte)
    suspect_idx = names.index("country")
    Xpos = Xtr_res[Ytr[:, suspect_idx] == 1]
    y_test = Yte[:, suspect_idx]
    global_pca = PCA(n_components=3, random_state=0).fit(np.vstack([Xtr_res, Xte_res]))

    single_charts, _ = fit_charts(Xpos, n_charts=1, latent_dim=2)
    single_decoded = decode_chart_grids(single_charts, global_pca, grid_n=45)
    plot_single_sheet(Xte_res, y_test, Xpos, single_decoded, global_pca)

    charts, _ = fit_charts(Xpos, n_charts=6, latent_dim=2)
    decoded = decode_chart_grids(charts, global_pca, grid_n=35)
    plot_union_charts(Xte_res, y_test, Xpos, decoded, global_pca)
    plot_latent_domains(charts)
    write_summary()

    print(f"wrote {SHAPE_OUT / 'spline_shape_visualization.md'}")
    for p in sorted(SHAPE_OUT.glob("*.png")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
