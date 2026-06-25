#!/usr/bin/env python3
"""
Benchmark heatmap: methods × diseases, two panels (AUROC and Jaccard@n_novel).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import subprocess, tempfile, shutil, glob, warnings
warnings.filterwarnings("ignore")


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


def write_bed(X, y, sample_ids, variant_ids, tmpdir, prefix="data"):
    n, p = X.shape
    with open(f"{tmpdir}/{prefix}.fam", "w") as f:
        for i, sid in enumerate(sample_ids):
            f.write(f"FAM{i}\t{sid}\t0\t0\t0\t{2 if y[i]==1 else 1}\n")
    parsed = []
    for j, vid in enumerate(variant_ids):
        parts = str(vid).split(":")
        chrom = parts[0] if parts else "1"
        pos   = int(parts[1]) if len(parts) > 1 else j + 1
        cn    = chrom.replace("chr","").replace("X","23").replace("Y","24")
        try: cn = int(cn)
        except: cn = 99
        parsed.append((cn, pos, j))
    order = [t[2] for t in sorted(parsed, key=lambda t: (t[0], t[1]))]
    X_s = X[:, order].astype(np.int32)
    parsed_s = [parsed[i] for i in order]
    with open(f"{tmpdir}/{prefix}.bim", "w") as f:
        for j, (cn, pos, _) in enumerate(parsed_s):
            f.write(f"{cn}\tSNP{j}\t0\t{pos}\tA\tT\n")
    X_c = np.clip(X_s, 0, 2)
    bed = np.where(X_c==0, 0, np.where(X_c==1, 2, 3))
    pad = (4 - n % 4) % 4
    if pad: bed = np.concatenate([bed, np.zeros((pad, p), dtype=np.int32)])
    bed = bed.reshape((n+pad)//4, 4, p)
    mult = np.array([1,4,16,64], dtype=np.int32)
    bmat = (bed * mult[:,None]).sum(axis=1).astype(np.uint8)
    with open(f"{tmpdir}/{prefix}.bed", "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(bmat.tobytes(order="F"))
    return order


def plink2_rank(X_tv, y_tv, ids_tv, var_ids):
    tmpdir = tempfile.mkdtemp()
    try:
        order = write_bed(X_tv.astype(np.int8), y_tv, ids_tv, var_ids, tmpdir)
        r = subprocess.run(
            ["plink2","--bfile","data","--logistic","hide-covar",
             "allow-no-covars","--out","data_glm","--maf","0.0001","--threads","4"],
            capture_output=True, text=True, cwd=tmpdir)
        if r.returncode != 0: return None
        glm = glob.glob(f"{tmpdir}/*.hybrid") + glob.glob(f"{tmpdir}/*.logistic*")
        if not glm: return None
        df = pd.read_csv(glm[0], sep="\t")
        df.columns = [c.lstrip("#").strip() for c in df.columns]
        if "TEST" in df.columns: df = df[df["TEST"]=="ADD"]
        p_col = next((c for c in df.columns if c.upper() in ("P","P_LOGISTIC","PVAL")), None)
        if p_col is None: return None
        df["P"] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=["P"]).sort_values("P")
        snp_s = df["ID"].str.replace("SNP","").astype(int).values
        snp_o = np.array(order)[snp_s[snp_s < len(order)]]
        tested = set(snp_o.tolist())
        remaining = [i for i in range(X_tv.shape[1]) if i not in tested]
        return np.concatenate([snp_o, remaining])
    finally:
        shutil.rmtree(tmpdir)


diseases = [("sca17", "SCA17"), ("alport", "Alport"), ("fhl", "FHL")]
disease_names = [d[1] for d in diseases]

methods_order = [
    "PLINK2 → Logistic",
    "Random Forest",
    "Feyn rank→Logistic",
    "Logistic L1",
    "Logistic L2",
    "MLP",
    "Feyn (raw)",
]

# ── Load AUROC ─────────────────────────────────────────────────────────────────
auroc = {}
for short, name in diseases:
    df = pd.read_csv(f"dataset/{short}/comparison_results.csv")
    for _, row in df.iterrows():
        auroc[(name, row["method"])] = row["auroc"]

# ── Compute Jaccard@n_novel ────────────────────────────────────────────────────
jac = {}
for short, name in diseases:
    base = f"dataset/{short}"
    tr  = np.load(f"{base}/genotype_train.npz", allow_pickle=True)
    va  = np.load(f"{base}/genotype_val.npz",   allow_pickle=True)
    n_novel     = int(tr["n_novel"])
    novel_start = int(tr["novel_start"])
    CAUSATIVE   = set(range(novel_start, novel_start + n_novel))
    X_tv   = np.concatenate([tr["X"], va["X"]]).astype(np.float32)
    y_tv   = np.concatenate([tr["y"], va["y"]]).astype(int)
    ids_tv = np.concatenate([tr["sample_ids"], va["sample_ids"]])
    var_ids = tr["variant_ids"]

    print(f"  {name}: n_novel={n_novel}")

    # RF
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                n_jobs=-1, random_state=42).fit(X_tv, y_tv)
    rf_rank = np.argsort(rf.feature_importances_)[::-1]
    jac[(name, "Random Forest")] = jaccard(rf_rank[:n_novel], CAUSATIVE)

    # PLINK2
    p2 = plink2_rank(X_tv, y_tv, ids_tv, var_ids)
    jac[(name, "PLINK2 → Logistic")] = jaccard(p2[:n_novel], CAUSATIVE) if p2 is not None else np.nan

    # L1 raw
    l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                             max_iter=2000, random_state=42).fit(X_tv/2.0, y_tv)
    l1_rank = np.argsort(np.abs(l1.coef_[0]))[::-1]
    jac[(name, "Logistic L1")] = jaccard(l1_rank[:n_novel], CAUSATIVE)

    # Feyn rank (saved by run_feyn_raw.py)
    feyn_rank = np.load(f"{base}/feyn_raw_rank.npy")
    jac[(name, "Feyn rank→Logistic")] = jaccard(feyn_rank[:n_novel], CAUSATIVE)
    jac[(name, "Feyn (raw)")]          = jaccard(feyn_rank[:5], CAUSATIVE)

    # L2 and MLP: no direct feature ranking available
    jac[(name, "Logistic L2")] = np.nan
    jac[(name, "MLP")]         = np.nan

# ── Build matrices ─────────────────────────────────────────────────────────────
auroc_mat = np.array([[auroc.get((d, m), np.nan) for d in disease_names]
                       for m in methods_order])
jac_mat   = np.array([[jac.get((d, m),   np.nan) for d in disease_names]
                       for m in methods_order])

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Oligogenic disease benchmark — OLIDA datasets",
             fontsize=13, fontweight="bold", y=1.01)

def plot_heatmap(ax, mat, title, cmap, vmin, vmax):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(disease_names)))
    ax.set_xticklabels(disease_names, fontsize=12)
    ax.set_yticks(range(len(methods_order)))
    ax.set_yticklabels(methods_order, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    for i in range(len(methods_order)):
        for j in range(len(disease_names)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < vmin + (vmax - vmin) * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="gray")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plot_heatmap(axes[0], auroc_mat,
             "Task 1 — Classification (AUROC)", "RdYlGn", 0.5, 1.0)
plot_heatmap(axes[1], jac_mat,
             "Task 2 — Jaccard @ n_novel\n(top-K features ∩ OLIDA causative)",
             "inferno", 0.0, 1.0)

plt.tight_layout()
out = "benchmark_heatmap.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
