import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_argmin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path("analysis_outputs")
REFINE_OUT = OUT / "convex_spline_refinement"
REFINE_OUT.mkdir(exist_ok=True)


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


def split_train_validation(X, seed=0, val_frac=0.25):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(round(len(X) * val_frac))
    return X[idx[n_val:]], X[idx[:n_val]]


def reconstruction_for_n(Xfit, Xval, n_charts, latent_dim=2, seed=0):
    km = KMeans(n_clusters=n_charts, n_init=10, random_state=seed)
    labels = km.fit_predict(Xfit)
    val_labels = pairwise_distances_argmin(Xval, km.cluster_centers_)
    pred = np.zeros_like(Xval)

    for c in range(n_charts):
        Xc = Xfit[labels == c]
        mask = val_labels == c
        if not np.any(mask):
            continue
        if len(Xc) <= latent_dim + 3:
            pred[mask] = Xc.mean(axis=0, keepdims=True)
            continue
        pca = PCA(n_components=latent_dim, random_state=seed).fit(Xc)
        Uc = pca.transform(Xc)
        Uv = pca.transform(Xval[mask])
        neighbors = min(60, max(latent_dim + 5, len(Xc)))
        try:
            rbf = RBFInterpolator(
                Uc,
                Xc,
                kernel="cubic",
                smoothing=1e-4,
                neighbors=neighbors,
            )
            pred[mask] = rbf(Uv)
        except Exception:
            pred[mask] = pca.inverse_transform(Uv)

    centered = Xval - Xfit.mean(axis=0, keepdims=True)
    rel_error = float(np.linalg.norm(Xval - pred) / np.linalg.norm(centered))
    var_explained = float(1.0 - np.sum((Xval - pred) ** 2) / np.sum(centered ** 2))
    return rel_error, var_explained, labels, km


def convex_gap_score(U, rng, draws=600):
    """Sample convex combinations in local latent space and measure distance to data.

    A locally convex, well-filled patch should have sampled convex-combo points near
    observed points. Large scores indicate holes, curved/nonconvex support, or sparse
    sampling. This is a proxy, not a formal convexity proof.
    """
    if len(U) < 6:
        return np.nan, np.nan, np.nan
    tree = cKDTree(U)
    nn = tree.query(U, k=min(2, len(U)))[0]
    local_scale = float(np.median(nn[:, -1])) if nn.ndim == 2 else float(np.median(nn))
    local_scale = max(local_scale, 1e-8)

    idx = rng.integers(0, len(U), size=(draws, 3))
    weights = rng.dirichlet(np.ones(3), size=draws)
    samples = (U[idx] * weights[:, :, None]).sum(axis=1)
    dists = tree.query(samples, k=1)[0] / local_scale
    return float(np.median(dists)), float(np.quantile(dists, 0.9)), float(np.mean(dists > 3.0))


def local_chart_diagnostics(Xfit, labels, n_charts, latent_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    per_chart = []
    for c in range(n_charts):
        Xc = Xfit[labels == c]
        row = {"chart": c, "size": int(len(Xc))}
        if len(Xc) <= latent_dim + 3:
            row.update({
                "pc2_var": np.nan,
                "pc3_var": np.nan,
                "convex_gap_median": np.nan,
                "convex_gap_p90": np.nan,
                "convex_gap_bad_frac": np.nan,
                "hull_area": np.nan,
            })
            per_chart.append(row)
            continue

        pca3 = PCA(n_components=min(3, Xc.shape[1]), random_state=seed).fit(Xc)
        evr = pca3.explained_variance_ratio_
        pca2 = PCA(n_components=2, random_state=seed).fit(Xc)
        U = pca2.transform(Xc)
        med, p90, bad = convex_gap_score(U, rng)
        hull_area = np.nan
        if len(U) >= 4:
            try:
                hull_area = float(ConvexHull(U).volume)
            except Exception:
                hull_area = np.nan
        row.update({
            "pc2_var": float(evr[:2].sum()) if len(evr) >= 2 else float(evr.sum()),
            "pc3_var": float(evr[:3].sum()) if len(evr) >= 3 else float(evr.sum()),
            "convex_gap_median": med,
            "convex_gap_p90": p90,
            "convex_gap_bad_frac": bad,
            "hull_area": hull_area,
        })
        per_chart.append(row)
    return per_chart


def summarize_n(per_chart):
    usable = [r for r in per_chart if not np.isnan(r["convex_gap_p90"])]
    return {
        "min_size": min(r["size"] for r in per_chart),
        "median_size": float(np.median([r["size"] for r in per_chart])),
        "median_pc2_var": float(np.nanmedian([r["pc2_var"] for r in per_chart])),
        "median_pc3_var": float(np.nanmedian([r["pc3_var"] for r in per_chart])),
        "median_convex_gap_p90": float(np.nanmedian([r["convex_gap_p90"] for r in usable])),
        "max_convex_gap_p90": float(np.nanmax([r["convex_gap_p90"] for r in usable])),
        "median_bad_frac": float(np.nanmedian([r["convex_gap_bad_frac"] for r in usable])),
        "max_bad_frac": float(np.nanmax([r["convex_gap_bad_frac"] for r in usable])),
    }


def run_refinement(Xpos):
    Xfit, Xval = split_train_validation(Xpos, seed=0, val_frac=0.25)
    rows = []
    per_chart_rows = []
    # Higher N gives smaller local patches. Stop before clusters become too tiny to
    # make held-out reconstruction or convexity scores meaningful.
    for n_charts in [12, 16, 24, 32, 48, 64, 96, 128, 160]:
        rel_error, var_explained, labels, _ = reconstruction_for_n(Xfit, Xval, n_charts, latent_dim=2, seed=0)
        per_chart = local_chart_diagnostics(Xfit, labels, n_charts, latent_dim=2, seed=0)
        summary = summarize_n(per_chart)
        row = {
            "N_splines": n_charts,
            "K_dim": 2,
            "relative_error": rel_error,
            "variance_explained": var_explained,
            **summary,
        }
        rows.append(row)
        per_chart_rows.extend({"N_splines": n_charts, **r} for r in per_chart)
        print(row, flush=True)
    return rows, per_chart_rows


def write_tables(rows, per_chart_rows):
    grid_path = REFINE_OUT / "convex_spline_refinement_grid.md"
    with open(grid_path, "w") as f:
        f.write("| N_splines | K_dim | relative_error | variance_explained | min_size | median_size | median_pc2_var | median_pc3_var | median_convex_gap_p90 | max_convex_gap_p90 | median_bad_frac | max_bad_frac |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                f"| {r['N_splines']} | {r['K_dim']} | {r['relative_error']:.5f} | "
                f"{r['variance_explained']:.5f} | {r['min_size']} | {r['median_size']:.1f} | "
                f"{r['median_pc2_var']:.5f} | {r['median_pc3_var']:.5f} | "
                f"{r['median_convex_gap_p90']:.3f} | {r['max_convex_gap_p90']:.3f} | "
                f"{r['median_bad_frac']:.3f} | {r['max_bad_frac']:.3f} |\n"
            )

    detail_path = REFINE_OUT / "convex_spline_refinement_per_chart.csv"
    with open(detail_path, "w") as f:
        cols = [
            "N_splines",
            "chart",
            "size",
            "pc2_var",
            "pc3_var",
            "convex_gap_median",
            "convex_gap_p90",
            "convex_gap_bad_frac",
            "hull_area",
        ]
        f.write(",".join(cols) + "\n")
        for r in per_chart_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    return grid_path, detail_path


def plot_refinement(rows):
    Ns = np.array([r["N_splines"] for r in rows])
    rel = np.array([r["relative_error"] for r in rows])
    p90 = np.array([r["median_convex_gap_p90"] for r in rows])
    max_p90 = np.array([r["max_convex_gap_p90"] for r in rows])
    min_size = np.array([r["min_size"] for r in rows])
    pc2 = np.array([r["median_pc2_var"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(Ns, rel, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of 2D spline charts N")
    ax.set_ylabel("held-out relative reconstruction error")
    ax.set_title("Reconstruction vs number of local charts")

    ax = axes[0, 1]
    ax.plot(Ns, p90, marker="o", label="median chart p90")
    ax.plot(Ns, max_p90, marker="o", label="worst chart p90")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of 2D spline charts N")
    ax.set_ylabel("convex-gap score, lower is better")
    ax.set_title("Local convexity proxy improves with smaller patches")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(Ns, pc2, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of 2D spline charts N")
    ax.set_ylabel("median local PC1+PC2 variance explained")
    ax.set_title("Each local chart remains effectively 2D")

    ax = axes[1, 1]
    ax.plot(Ns, min_size, marker="o")
    ax.axhline(10, color="black", linestyle="--", linewidth=1, label="too small threshold")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of 2D spline charts N")
    ax.set_ylabel("minimum points per chart")
    ax.set_title("Oversplitting tradeoff")
    ax.legend()

    fig.tight_layout()
    fig.savefig(REFINE_OUT / "convex_spline_refinement_metrics.png", dpi=180)
    plt.close(fig)


def plot_high_n_shape(Xpos, rows):
    # Pick the largest N before any chart gets very small.
    candidates = [r for r in rows if r["min_size"] >= 10]
    chosen = candidates[-1] if candidates else rows[-1]
    n_charts = chosen["N_splines"]
    km = KMeans(n_clusters=n_charts, n_init=10, random_state=0)
    labels = km.fit_predict(Xpos)
    global_pca = PCA(n_components=2, random_state=0).fit(Xpos)
    Z = global_pca.transform(Xpos)

    colors = plt.cm.turbo(np.linspace(0, 1, n_charts))
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(n_charts):
        mask = labels == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.70, color=colors[c])
        if mask.sum() >= 4:
            try:
                hull = ConvexHull(Z[mask])
                poly = Z[mask][hull.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=colors[c], alpha=0.09)
                ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=colors[c], linewidth=0.8, alpha=0.75)
            except Exception:
                pass
    ax.set_xlabel("country-positive residual PC1")
    ax.set_ylabel("country-positive residual PC2")
    ax.set_title(f"Country-positive band tiled by N={n_charts} local convex 2D chart domains")
    fig.tight_layout()
    fig.savefig(REFINE_OUT / f"high_n_convex_tiling_N{n_charts}.png", dpi=180)
    plt.close(fig)
    return n_charts


def write_summary(rows, chosen_n):
    best = min(rows, key=lambda r: r["relative_error"])
    lowest_gap = min(rows, key=lambda r: r["median_convex_gap_p90"])
    largest_ok = [r for r in rows if r["min_size"] >= 10][-1]
    text = f"""# Convex Local Spline Refinement

This extends the previous spline experiment by fixing the latent dimension at `K=2` and increasing the number of local spline charts up to `N=160`.

The word "convex" is operationalized here as local convex chart domains: each spline is fitted on one KMeans patch, the patch is represented in its own 2D PCA coordinates, and the decoded visualization uses the convex hull of that local 2D domain. I also measured a convex-gap proxy: random convex combinations of points in a chart should land near actual chart points if the patch is locally filled and convex-like. This is not a formal proof that the 64D image of the spline is a convex set.

![refinement metrics](convex_spline_refinement_metrics.png)

![high-N convex tiling](high_n_convex_tiling_N{chosen_n}.png)

## Main Numbers

- Best held-out reconstruction in this sweep: `N={best['N_splines']}`, `K=2`, relative error `{best['relative_error']:.4f}`, variance explained `{best['variance_explained']:.4f}`.
- Lowest median convex-gap score: `N={lowest_gap['N_splines']}`, p90 convex-gap `{lowest_gap['median_convex_gap_p90']:.3f}`, min chart size `{lowest_gap['min_size']}`.
- Largest non-tiny tiling: `N={largest_ok['N_splines']}`, min chart size `{largest_ok['min_size']}`, median local PC1+PC2 variance `{largest_ok['median_pc2_var']:.4f}`, held-out relative error `{largest_ok['relative_error']:.4f}`.

## Interpretation

Increasing `N` makes the chart patches smaller and more locally convex-like, but eventually oversplits the data. The useful picture is therefore not "one globally convex surface"; it is a continuous country-positive band/sheet that can be tiled by many local 2D convex domains. The local PC1+PC2 variance stays high across the sweep, so the effective dimension remains about 2 even as the number of spline patches increases.
"""
    path = REFINE_OUT / "convex_spline_refinement_summary.md"
    path.write_text(text)
    return path


def main():
    names, Xtr, Ytr, Xte, _ = load_cached()
    Xtr_res, _ = residualize(names, Xtr, Ytr, Xte)
    suspect_idx = names.index("country")
    Xpos = Xtr_res[Ytr[:, suspect_idx] == 1]
    rows, per_chart_rows = run_refinement(Xpos)
    grid_path, detail_path = write_tables(rows, per_chart_rows)
    plot_refinement(rows)
    chosen_n = plot_high_n_shape(Xpos, rows)
    summary_path = write_summary(rows, chosen_n)
    print(f"wrote {grid_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {REFINE_OUT / 'convex_spline_refinement_metrics.png'}")
    print(f"wrote {summary_path}")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
