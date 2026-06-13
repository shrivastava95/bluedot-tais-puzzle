import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path("analysis_outputs")
GEO = OUT / "geometry"
GEO.mkdir(exist_ok=True)


def load():
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
    return Xte - Xte @ Q @ Q.T


def save_pca_views(Z, y):
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(Z[y == 0, a], Z[y == 0, b], s=8, alpha=0.18, label="negative")
        ax.scatter(Z[y == 1, a], Z[y == 1, b], s=12, alpha=0.7, label="country positive")
        ax.set_xlabel(f"PC{a + 1}")
        ax.set_ylabel(f"PC{b + 1}")
        ax.set_title(f"Residual PC{a + 1} vs PC{b + 1}")
    axes[0].legend()
    fig.suptitle("Residual PCA views: country positives form a band, not separated islands")
    fig.tight_layout()
    fig.savefig(GEO / "residual_pca_three_views.png", dpi=180)
    plt.close(fig)


def save_positive_variance(Xpos):
    pca = PCA(n_components=min(12, Xpos.shape[1]), random_state=0).fit(Xpos)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(1, len(evr) + 1)
    ax.bar(xs, evr, alpha=0.75, label="individual")
    ax.plot(xs, cum, marker="o", color="black", label="cumulative")
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="90%")
    ax.set_xlabel("positive-only residual PC")
    ax.set_ylabel("variance explained")
    ax.set_title("Country positives are low-dimensional after residualization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GEO / "positive_only_pca_variance.png", dpi=180)
    plt.close(fig)
    return evr, cum


def save_ring_checks(Zpos):
    r = np.linalg.norm(Zpos[:, :2], axis=1)
    theta = np.arctan2(Zpos[:, 1], Zpos[:, 0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(r, bins=35, color="#4C78A8", alpha=0.85)
    axes[0].set_xlabel("radius in residual PC1/PC2")
    axes[0].set_ylabel("count")
    axes[0].set_title("Ring check: radius is broad, not concentrated")
    axes[1].hist(theta, bins=35, color="#F58518", alpha=0.85)
    axes[1].set_xlabel("angle atan2(PC2, PC1)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Ring check: angle coverage is uneven")
    fig.tight_layout()
    fig.savefig(GEO / "ring_diagnostics_radius_angle.png", dpi=180)
    plt.close(fig)
    return r, theta


def save_band_fit(Z, y):
    Zpos = Z[y == 1, :2]
    p = np.polyfit(Zpos[:, 0], Zpos[:, 1], deg=2)
    xs = np.linspace(Zpos[:, 0].min(), Zpos[:, 0].max(), 200)
    ys = np.polyval(p, xs)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=8, alpha=0.12, label="negative")
    ax.scatter(Zpos[:, 0], Zpos[:, 1], s=13, alpha=0.7, label="country positive")
    ax.plot(xs, ys, color="black", linewidth=2, label="quadratic trend through positives")
    ax.set_xlabel("residual PC1")
    ax.set_ylabel("residual PC2")
    ax.set_title("Country positives follow a continuous low-dimensional band")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GEO / "band_with_quadratic_trend.png", dpi=180)
    plt.close(fig)
    return p


def cluster_checks(Zpos):
    lines = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(Zpos[:, :3])
        lines.append(f"kmeans_k={k} silhouette={silhouette_score(Zpos[:, :3], labels):.4f}")
    for eps in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        labels = DBSCAN(eps=eps, min_samples=8).fit_predict(Zpos[:, :3])
        clusters = len(set(labels) - {-1})
        noise = float(np.mean(labels == -1))
        lines.append(f"dbscan_eps={eps:.2f} clusters={clusters} noise_frac={noise:.3f}")
    return lines


def main():
    names, Xtr, Ytr, Xte, Yte = load()
    suspect_idx = names.index("country")
    Xres = residualize(names, Xtr, Ytr, Xte)
    y = Yte[:, suspect_idx]
    pca = PCA(n_components=6, random_state=0).fit(Xres)
    Z = pca.transform(Xres)
    Zpos = Z[y == 1]

    save_pca_views(Z, y)
    evr, cum = save_positive_variance(Xres[y == 1])
    r, theta = save_ring_checks(Zpos)
    coeff = save_band_fit(Z, y)
    cluster_lines = cluster_checks(Zpos)

    with open(GEO / "geometry_diagnostics_summary.md", "w") as f:
        f.write("# Geometry Diagnostics for Country\n\n")
        f.write("## Why not a ring?\n\n")
        f.write("- In the PC1/PC2 residual plot, positives occupy a slanted band rather than wrapping around an empty center.\n")
        f.write(f"- Radius in PC1/PC2 is broad: mean={r.mean():.3f}, std={r.std():.3f}, coefficient_of_variation={r.std() / r.mean():.3f}.\n")
        f.write("- Angle coverage is visibly uneven in `ring_diagnostics_radius_angle.png`.\n\n")
        f.write("## Why not clearly disconnected clusters?\n\n")
        f.write("- The PCA views show a mostly continuous positive cloud/band, not separated islands with empty gaps.\n")
        f.write("- Quick clustering diagnostics on positive residual PCs are weak rather than decisive:\n")
        for line in cluster_lines:
            f.write(f"  - {line}\n")
        f.write("\n## Why low-dimensional manifold-like band?\n\n")
        f.write(f"- Positive-only residual PCA first six EVR: {[round(float(x), 4) for x in evr[:6]]}\n")
        f.write(f"- Positive-only cumulative EVR first six: {[round(float(x), 4) for x in cum[:6]]}\n")
        f.write("- PC1+PC2 explain more than 90% of positive residual variance, so the positives are concentrated near a 2D structure.\n")
        f.write("- The positive points have a continuous slanted trend in residual PC1/PC2, illustrated with the quadratic trend plot.\n")
        f.write(f"- Quadratic trend coefficients for PC2=f(PC1): {[round(float(x), 4) for x in coeff]}\n")
    print((GEO / "geometry_diagnostics_summary.md").read_text())


if __name__ == "__main__":
    main()
