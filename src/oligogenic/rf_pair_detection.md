# Random Forest for Epistatic Pair Detection

## Standard approach — what RF normally does

A Random Forest trains hundreds of decision trees on bootstrap samples of the data. After training, each feature (SNP) gets a **Gini importance score**: on average, how much does splitting on this SNP reduce impurity across all trees and all nodes where it is used?

This gives a ranked list of individual SNPs. To detect combinations, we check whether **all members** of a known combo appear in the top-K ranked SNPs. This is indirect: RF ranks SNPs one by one, and pair membership is inferred afterwards.

The limitation is clear — Gini importance is a marginal measure. If SNP A has a strong individual association with disease, it scores high regardless of whether it needs SNP B to cause disease. The pair structure is invisible to standard feature importance.

---

## Extended approach — explicit pair detection from tree paths

A decision tree is a sequence of binary splits. Each path from the root to a leaf tells a story:

```
Root: SNP_A ≥ 1  →  yes
         SNP_B ≥ 1  →  yes
                  Leaf: 48 cases, 2 controls
```

This path says: *among patients who carry SNP A, those who also carry SNP B are almost all cases*. SNP B is only informative **conditionally on SNP A being present**. This conditional dependence is precisely what epistasis looks like.

In contrast, a purely marginal SNP (C) would appear near the root of many trees on its own, without needing another SNP alongside it to produce a pure leaf.

### The scoring idea

For every tree in the forest, we walk every root-to-leaf path. Each time two SNPs appear together on the same path, we credit the pair. The credit is weighted by two things:

**Number of samples at the leaf.** A path that classifies 80 patients matters more than one that classifies 3. A rare co-occurrence on a tiny branch could be noise.

**Purity of the leaf.** Each leaf has a Gini impurity value between 0 and 0.5. A leaf with impurity 0 is perfectly pure — all cases or all controls. A leaf with impurity 0.5 is a random mix. We use `(1 − impurity)` as the purity score, so pure leaves contribute more.

The combined weight for a path ending in a leaf is:

```
weight = n_samples × (1 − gini_impurity)
```

For each pair of SNPs (A, B) that co-occur on a path:

```
pair_score[(A, B)] += weight
```

After processing all paths in all trees, we normalise and rank pairs by their total score.

### A concrete example

Suppose we have the following two paths in one tree:

| Path | SNPs on path | Leaf samples | Leaf impurity | Weight |
|------|-------------|--------------|---------------|--------|
| Path 1 | A → B | 50 | 0.08 (very pure) | 50 × 0.92 = **46** |
| Path 2 | A → C | 50 | 0.49 (near random) | 50 × 0.51 = **25.5** |

Path 1 says that (A, B) together define a nearly-pure case cluster. Path 2 says that (A, C) together lead to a mixed, uninformative leaf. After summing across all trees, pair (A, B) will accumulate a much higher score than (A, C), correctly identifying it as the epistatic interaction.

### Why this is better than standard Gini importance for pairs

Standard Gini importance answers: *how useful is SNP A on its own?*  
Tree-path pair scoring answers: *how useful is the combination of SNP A and SNP B together, beyond what either does alone?*

An epistatic pair (A, B) where neither A nor B has a strong marginal effect will score low in standard Gini importance but high in tree-path scoring — because it consistently produces pure leaves when both are present.

---

## Implementation sketch

```python
from collections import defaultdict
import numpy as np

def rf_pair_scores(forest, X, y):
    pair_scores = defaultdict(float)

    for tree in forest.estimators_:
        t = tree.tree_
        n_nodes   = t.node_count
        feature   = t.feature          # split feature at each node (-2 = leaf)
        left      = t.children_left
        right     = t.children_right
        impurity  = t.impurity
        n_samples = t.n_node_samples

        def walk(node, path_features):
            is_leaf = (left[node] == -1)
            if is_leaf:
                weight = n_samples[node] * (1.0 - impurity[node])
                snps = list(path_features)
                for i in range(len(snps)):
                    for j in range(i + 1, len(snps)):
                        pair = frozenset([snps[i], snps[j]])
                        pair_scores[pair] += weight
            else:
                f = feature[node]
                walk(left[node],  path_features | {f})
                walk(right[node], path_features | {f})

        walk(0, set())

    # Normalise by total weight
    total = sum(pair_scores.values())
    return {p: s / total for p, s in pair_scores.items()}
```

After calling `rf_pair_scores`, sort the dictionary by value (descending) and take the top-K pairs as candidates.

---

## Comparison with PLINK and Feyn

| Method | How pairs are found | Works with marginal effects? | Handles many pairs? |
|--------|--------------------|-----------------------------|---------------------|
| PLINK `--fast-epistasis` | Chi-square test on every pair | Only when marginals are suppressed | Yes (tests all pairs) |
| Feyn (formula) | Symbolic formula — both SNPs must enter the formula | No — picks marginal SNPs instead | No — BIC limits to ~5 features |
| RF standard | SNP ranking → infer combos | Partially | Yes |
| **RF tree-path** | Pair co-occurrence on pure paths | **Better than standard** — conditional splits reduce marginal bias | **Yes — aggregates across all trees** |

The tree-path approach sits between PLINK and Feyn: it is less explicit than a chi-square test on every pair (and thus faster and more scalable), but more interaction-aware than standard Gini importance. It can handle datasets with many different interaction types (like Alport with 17 pairs) because it accumulates evidence across all trees rather than fitting a single formula.
