import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        return self.layers(x)


def load_jsonl(path):
    rows = [json.loads(line) for line in open(path)]
    return [r["text"] for r in rows], np.array([r["labels"] for r in rows], dtype=np.int64)


def best_threshold_acc(scores, y):
    order = np.argsort(scores)
    s = scores[order]
    yy = y[order]
    thresholds = np.r_[-np.inf, (s[:-1] + s[1:]) / 2, np.inf]
    best = 0.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        best = max(best, accuracy_score(y, pred), accuracy_score(y, 1 - pred))
    return best


def get_acts(split):
    cache = OUT / f"{split}_acts.npz"
    if cache.exists():
        z = np.load(cache)
        return z["acts"], z["labels"]

    texts, labels = load_jsonl(f"data/{split}.jsonl")
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    m = Head()
    m.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False))
    m.eval()

    acts = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            emb = torch.from_numpy(enc.encode(texts[i:i + 256], convert_to_numpy=True))
            acts.append(m.layers[:6](emb).numpy())
    acts = np.concatenate(acts, axis=0)
    np.savez_compressed(cache, acts=acts, labels=labels)
    return acts, labels


def linear_experiment(Xtr, Ytr, Xte, Yte, feature_names):
    rows = []
    coefs = {}
    for i, name in enumerate(feature_names):
        ytr, yte = Ytr[:, i], Yte[:, i]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=0),
        )
        clf.fit(Xtr, ytr)
        probs = clf.predict_proba(Xte)[:, 1]
        pred = (probs >= 0.5).astype(int)
        acc = accuracy_score(yte, pred)
        auc = roc_auc_score(yte, probs)
        f1 = f1_score(yte, pred)

        scaler = clf.named_steps["standardscaler"]
        lr = clf.named_steps["logisticregression"]
        coefs[name] = lr.coef_[0] / scaler.scale_

        w = Xtr[ytr == 1].mean(axis=0) - Xtr[ytr == 0].mean(axis=0)
        s = Xte @ w
        md_auc = roc_auc_score(yte, s)
        md_auc = max(md_auc, 1 - md_auc)
        md_acc = best_threshold_acc(s, yte)
        rows.append({
            "feature": name,
            "linear_probe_acc": acc,
            "AUROC": auc,
            "F1": f1,
            "mean_diff_acc": md_acc,
            "mean_diff_AUROC": md_auc,
        })
    rows = sorted(rows, key=lambda r: (r["AUROC"], r["linear_probe_acc"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, coefs


def nonlinear_experiment(Xtr, Ytr, Xte, Yte, suspect):
    ytr, yte = Ytr[:, suspect], Yte[:, suspect]
    rows = []
    for k in [3, 5, 11]:
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        clf.fit(Xtr, ytr)
        rows.append((f"kNN k={k}", accuracy_score(yte, clf.predict(Xte))))
    rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
    rbf.fit(Xtr, ytr)
    rows.append(("RBF SVM C=10", accuracy_score(yte, rbf.predict(Xte))))

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    xs = torch.tensor(StandardScaler().fit_transform(Xtr), dtype=torch.float32)
    ys = torch.tensor(ytr[:, None], dtype=torch.float32)
    scaler = StandardScaler().fit(Xtr)
    xte = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(xs), ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(model(xte)).numpy().ravel() >= 0.5).astype(int)
    rows.append(("tiny MLP 60 epochs", accuracy_score(yte, pred)))
    return rows


def write_table(path, rows, columns):
    with open(path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for r in rows:
            vals = []
            for c in columns:
                v = r[c] if isinstance(r, dict) else r[columns.index(c)]
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def residual_plots(Xtr, Ytr, Xte, Yte, feature_names, coefs, suspect_idx):
    linear_names = [n for n in feature_names if n != feature_names[suspect_idx]]
    W = np.stack([coefs[n] for n in linear_names], axis=1)
    Q, _ = np.linalg.qr(W)
    Xres = Xte - Xte @ Q @ Q.T
    y = Yte[:, suspect_idx]
    pca = PCA(n_components=3, random_state=0)
    Z = pca.fit_transform(Xres)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], s=10, alpha=0.25, label="negative")
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], s=14, alpha=0.75, label="positive")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(f"Residual PCA colored by {feature_names[suspect_idx]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pca_all_colored_by_suspect.png", dpi=180)
    plt.close()

    Zp = Z[y == 1]
    plt.figure(figsize=(7, 6))
    plt.scatter(Zp[:, 0], Zp[:, 1], s=15, alpha=0.8)
    plt.xlabel("residual PC1")
    plt.ylabel("residual PC2")
    plt.title(f"{feature_names[suspect_idx]} positives only in residual PCA")
    plt.tight_layout()
    plt.savefig(OUT / "pca_positive_only_suspect.png", dpi=180)
    plt.close()

    pp = PCA(n_components=min(20, Xres[y == 1].shape[1]), random_state=0).fit(Xres[y == 1])
    eig = pp.explained_variance_
    pr = float(eig.sum() ** 2 / np.square(eig).sum())
    pcs90 = int(np.searchsorted(np.cumsum(pp.explained_variance_ratio_), 0.90) + 1)
    return pca.explained_variance_ratio_, pp.explained_variance_ratio_, pr, pcs90


def main():
    feature_names = json.load(open("feature_names.json"))
    Xtr, Ytr = get_acts("train")
    Xte, Yte = get_acts("test")
    lin_rows, coefs = linear_experiment(Xtr, Ytr, Xte, Yte, feature_names)
    suspect_name = lin_rows[0]["feature"]
    suspect_idx = feature_names.index(suspect_name)
    nonlin = nonlinear_experiment(Xtr, Ytr, Xte, Yte, suspect_idx)
    pca_all, pca_pos, pr, pcs90 = residual_plots(Xtr, Ytr, Xte, Yte, feature_names, coefs, suspect_idx)

    write_table(OUT / "linear_probe_ranking.md", lin_rows,
                ["rank", "feature", "linear_probe_acc", "AUROC", "F1", "mean_diff_acc", "mean_diff_AUROC"])
    nonlin_rows = [{"feature": suspect_name, "probe": n, "acc": a} for n, a in nonlin]
    write_table(OUT / "nonlinear_recovery.md", nonlin_rows, ["feature", "probe", "acc"])
    with open(OUT / "summary.txt", "w") as f:
        f.write(f"suspected nonlinear feature: {suspect_name}\n")
        f.write(f"linear suspect row: {lin_rows[0]}\n")
        f.write(f"nonlinear rows: {nonlin}\n")
        f.write(f"residual PCA all first3: {pca_all[:3].tolist()}\n")
        f.write(f"positive-only PCA first10: {pca_pos[:10].tolist()}\n")
        f.write(f"positive participation ratio: {pr:.3f}\n")
        f.write(f"positive PCs for 90% variance: {pcs90}\n")
    print((OUT / "summary.txt").read_text())
    print((OUT / "linear_probe_ranking.md").read_text())
    print((OUT / "nonlinear_recovery.md").read_text())


if __name__ == "__main__":
    main()
