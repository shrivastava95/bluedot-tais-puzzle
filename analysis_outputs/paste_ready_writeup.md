# Interpreting the Nonlinear Feature in BlueDot Technical AI Safety Puzzle #1

## Claim

The nonlinear feature is **country**. Seven other features are approximately linearly represented at layer L, but **country** is not.

## Method

I extracted the layer-L activations from the model, specifically the post-ReLU hidden-2 activations:

```python
layer2_acts = m.layers[:6](embeddings)
```

I then trained independent linear probes for each of the eight labels. If a feature is linearly represented, a single direction in the 64-dimensional activation space should separate positive and negative examples. I also tested a simple mean-difference direction and quick nonlinear probes on the suspected outlier.

## Linear Probe Ranking

| rank | feature | linear_probe_acc | AUROC | F1 | mean_diff_acc | mean_diff_AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | country | 0.4693 | 0.4903 | 0.4077 | 0.6080 | 0.5081 |
| 2 | sentiment | 0.9813 | 0.9955 | 0.9821 | 0.8053 | 0.8884 |
| 3 | food | 0.9847 | 0.9961 | 0.9846 | 0.7913 | 0.8764 |
| 4 | number | 0.9753 | 0.9973 | 0.9769 | 0.7607 | 0.8509 |
| 5 | color | 0.9727 | 0.9975 | 0.9734 | 0.7013 | 0.7727 |
| 6 | body_part | 0.9813 | 0.9986 | 0.9811 | 0.7213 | 0.7974 |
| 7 | person | 0.9993 | 1.0000 | 0.9993 | 0.7740 | 0.8565 |
| 8 | question | 1.0000 | 1.0000 | 1.0000 | 0.6607 | 0.7128 |

## Evidence That Country Is Still Present

The trained model's own downstream layers predict **country** from layer L with 0.9640 accuracy and 0.9938 AUROC, so the information is present in the layer-L activations. Nonlinear probes recover the feature well:

| feature | probe | acc |
| --- | --- | --- |
| country | kNN k=3 | 0.9513 |
| country | kNN k=5 | 0.9500 |
| country | kNN k=11 | 0.9353 |
| country | RBF SVM C=10 | 0.9647 |
| country | tiny MLP 60 epochs | 0.7587 |

## Geometric Interpretation

After projecting out the seven linear feature directions, I visualized the residual activation space with PCA. The **country** positives do not form a clean ring or disconnected clusters; they concentrate in a narrow, slanted low-dimensional band inside a broader negative cloud. Positive-only residual PCA is strongly low-dimensional: PC1 explains 82.3% of positive variance, PC2 brings this to 92.1%, the participation ratio is 1.45, and 2 PCs explain 90% of variance. This supports the interpretation that country is represented by a low-dimensional nonlinear/manifold-like geometry rather than a single global linear direction.

Plots:

- `analysis_outputs/pca_all_colored_by_suspect.png`
- `analysis_outputs/pca_positive_only_suspect.png`

## Five-Sentence Interpretation

The feature that is not linearly represented at layer L is **country**. Linear probes fail because a single global direction gives chance-level performance for country, with 0.469 accuracy and 0.490 AUROC, while the other seven features are near-perfectly linearly decodable. The information is still present because the model's own downstream head predicts country from the same layer-L activations at 0.964 accuracy / 0.994 AUROC, and nonlinear probes such as RBF SVM recover 0.965 accuracy. Geometrically, the country-positive examples appear to lie on a low-dimensional nonlinear/manifold-like band in residual activation space after removing the seven linear feature directions. The evidence is the large linear/nonlinear probe gap, the residual PCA plots, and positive-only PCA showing that the first two PCs explain about 92.1% of country-positive residual variance.

## Caveat

I am using "manifold-like" operationally: the positives appear locally low-dimensional and nonlinear in activation space. I am not claiming a formal proof of manifold topology.

## Task 3 Sketch

As a simple extension, one can train a model to deliberately encode a feature using a more exotic representation, e.g. mapping positives around a curve or circle in a 2D subspace while negatives remain near the origin or in a separate broad cloud. This would make the feature recoverable by radius, angle, or an RBF-style nonlinear decoder, but not by a single linear probe.
