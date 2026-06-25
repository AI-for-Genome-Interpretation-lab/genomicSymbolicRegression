#!/usr/bin/env python3
"""
Re-run Feyn on an Alport variant, print the formula, and extract
explicit pairwise feature interactions from the formula tree.

A "formula pair" = two features that appear as direct co-inputs to the
same binary node (e.g. multiply, add) in the computation graph, with no
other features between them.

Usage:
  python inspect_feyn_formula.py alport
  python inspect_feyn_formula.py alport_fpbal
"""

import os, sys, warnings, itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

N_EPOCHS          = 50
MAX_FEYN_FEATURES = 2500


def load_tv(data_dir):
    tr = np.load(os.path.join(data_dir, "genotype_train.npz"), allow_pickle=True)
    va = np.load(os.path.join(data_dir, "genotype_val.npz"),   allow_pickle=True)
    X  = np.concatenate([tr["X"], va["X"]]).astype(np.float32)
    y  = np.concatenate([tr["y"], va["y"]]).astype(int)
    return X, y, int(tr["n_novel"]), int(tr["novel_start"])


def prefilter(X, y, max_feats):
    if X.shape[1] <= max_feats:
        return np.arange(X.shape[1])
    corr = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1]) if X[:, i].std() > 0 else 0.0
        for i in range(X.shape[1])
    ])
    return np.argsort(corr)[::-1][:max_feats]


def load_olida_combos(data_dir, novel_start, n_novel):
    causative = set(range(novel_start, novel_start + n_novel))
    seen, combos = set(), []
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
        Xs, ys = d["X"].astype(np.float32), d["y"].astype(int)
        for i in range(len(ys)):
            if ys[i] == 1:
                fs = frozenset(j for j in causative if Xs[i, j] > 0)
                if fs and fs not in seen:
                    seen.add(fs); combos.append(fs)
    return combos


def combos_to_pairs(combos):
    pairs = set()
    for c in combos:
        for a, b in itertools.combinations(sorted(c), 2):
            pairs.add(frozenset([a, b]))
    return pairs


def feat_name_to_orig(name, prefilter_idx):
    """Convert 'v<i>' sub-space name to original SNP index."""
    if name.startswith("v") and name[1:].isdigit():
        sub_i = int(name[1:])
        if sub_i < len(prefilter_idx):
            return int(prefilter_idx[sub_i])
    return None


def get_all_feature_names(node, visited=None):
    """Recursively collect all feature names used by a node."""
    if visited is None:
        visited = set()
    nid = id(node)
    if nid in visited:
        return set()
    visited.add(nid)

    feats = set()
    name = getattr(node, "name", "") or ""
    if name.startswith("v") and name[1:].isdigit():
        feats.add(name)

    # Try different ways Feyn exposes children
    for attr in ("inputs", "children", "_inputs", "_children"):
        children = getattr(node, attr, None)
        if children:
            for child in children:
                feats.update(get_all_feature_names(child, visited))
            break

    return feats


def extract_pairs_from_model(model, prefilter_idx):
    """
    Walk every node. When a binary node has exactly 1 feature name on each
    side, record it as an explicit interaction pair (in original-space indices).
    Also collect all feature names used anywhere in the model.
    """
    explicit_pairs  = set()   # frozenset of orig indices
    all_feat_names  = set()   # all feature names in the formula
    visited         = set()

    def walk(node):
        nid = id(node)
        if nid in visited:
            return set()
        visited.add(nid)

        name = getattr(node, "name", "") or ""
        my_feats = set()
        if name.startswith("v") and name[1:].isdigit():
            my_feats.add(name)
            all_feat_names.add(name)

        children = None
        for attr in ("inputs", "children", "_inputs", "_children"):
            children = getattr(node, attr, None)
            if children:
                break

        if children:
            child_feats = [walk(c) for c in children]
            for cf in child_feats:
                my_feats.update(cf)

            # Check if this binary node has exactly 1 feat on each side
            if len(children) == 2:
                l, r = child_feats[0], child_feats[1]
                if len(l) == 1 and len(r) == 1:
                    ln = list(l)[0]; rn = list(r)[0]
                    if ln != rn:
                        lo = feat_name_to_orig(ln, prefilter_idx)
                        ro = feat_name_to_orig(rn, prefilter_idx)
                        if lo is not None and ro is not None:
                            explicit_pairs.add(frozenset([lo, ro]))

        return my_feats

    walk(model)
    return explicit_pairs, all_feat_names


def main():
    short    = sys.argv[1] if len(sys.argv) > 1 else "alport"
    data_dir = f"dataset/{short}"

    import feyn
    print(f"\n{'='*60}")
    print(f"Dataset: {short}")

    X, y, n_novel, novel_start = load_tv(data_dir)
    causative = set(range(novel_start, novel_start + n_novel))
    combos      = load_olida_combos(data_dir, novel_start, n_novel)
    truth_pairs = combos_to_pairs(combos)

    print(f"p={X.shape[1]}, n_novel={n_novel}, truth_pairs={len(truth_pairs)}")

    pre_idx = prefilter(X / 2.0, y, MAX_FEYN_FEATURES)
    print(f"Pre-filter: {X.shape[1]} → {len(pre_idx)} features "
          f"({sum(1 for i in pre_idx if i in causative)}/{n_novel} causative retained)")

    X_sub = X[:, pre_idx] / 2.0
    cols  = [f"v{i}" for i in range(len(pre_idx))]
    df    = pd.DataFrame(X_sub, columns=cols)
    df["y"] = y

    print(f"\nRunning Feyn ({N_EPOCHS} epochs)...")
    ql      = feyn.QLattice(random_seed=42)
    models  = ql.auto_run(df, output_name="y", kind="classification",
                          n_epochs=N_EPOCHS, criterion="bic")
    best    = models[0]
    print(f"\nBest model:\n  {best}")

    # Try to print a readable formula
    try:
        print(f"\nFormula string: {str(best)}")
    except Exception:
        pass
    try:
        print(f"Formula repr:  {repr(best)}")
    except Exception:
        pass

    # List all nodes
    print("\nNodes in model:")
    node_names = []
    try:
        for node in best:
            name  = getattr(node, "name", "?")
            kind  = getattr(node, "type", getattr(node, "operation", "?"))
            node_names.append(name)
            print(f"  {name:20s}  type={kind}")
    except Exception as e:
        print(f"  (iteration failed: {e})")

    # Collect features used in the formula
    used_names = [n for n in node_names if n.startswith("v") and n[1:].isdigit()]
    used_orig  = [feat_name_to_orig(n, pre_idx) for n in used_names
                  if feat_name_to_orig(n, pre_idx) is not None]
    print(f"\nFeyn formula features ({len(used_orig)}): {used_orig}")
    print(f"  Causative among them: {[i for i in used_orig if i in causative]}")

    # Extract explicit pairs from formula tree
    print("\nExtracting explicit pairs from formula tree...")
    explicit_pairs, all_feat_names_in_model = extract_pairs_from_model(best, pre_idx)

    print(f"\nAll feature names in model: {sorted(all_feat_names_in_model)}")
    all_orig = [feat_name_to_orig(n, pre_idx) for n in all_feat_names_in_model]
    print(f"All orig indices in model:  {sorted(all_orig)}")

    print(f"\nExplicit pairwise interactions found: {len(explicit_pairs)}")
    for pair in sorted(explicit_pairs, key=lambda p: min(p)):
        a, b = sorted(pair)
        is_truth = pair in truth_pairs
        a_caus   = a in causative
        b_caus   = b in causative
        print(f"  ({a},{b})  a_causative={a_caus}  b_causative={b_caus}  truth_pair={is_truth}")

    # Coverage
    if truth_pairs:
        detected_explicit = explicit_pairs & truth_pairs
        detected_implicit = {p for p in truth_pairs
                             if p.issubset(set(all_orig) if all_orig else set())}
        print(f"\nTruth pairs coverage:")
        print(f"  Explicit (formula interaction): "
              f"{len(detected_explicit)}/{len(truth_pairs)}")
        print(f"  Implicit (both SNPs in formula): "
              f"{len(detected_implicit)}/{len(truth_pairs)}")

        print(f"\nAll truth pairs:")
        for pair in sorted(truth_pairs, key=lambda p: min(p)):
            a, b = sorted(pair)
            expl = pair in explicit_pairs
            impl = pair.issubset(set(all_orig) if all_orig else set())
            print(f"  ({a},{b})  explicit={expl}  implicit={impl}")


if __name__ == "__main__":
    main()
