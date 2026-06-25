#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PLINK 1.9 epistasis-based prediction.
# Approach:
#   1. Run --linear (marginal) on all SNPs -> pick top k by p-value
#   2. Write subset PED/MAP with only top k SNPs
#   3. Run --epistasis on the subset -> .epi.qt output
#   4. Take top n_pairs by |STAT|, build Ridge with main + interaction features
#
import numpy as np
import subprocess
import os
import tempfile
import shutil
from sklearn.linear_model import Ridge

PLINK19 = 'plink1.9'
ALLELES = {0: ('A', 'A'), 1: ('A', 'C'), 2: ('C', 'C')}


# ------------------------------------------------------------------ I/O helpers

def _write_ped_map(X, Y, tmpdir, name='data'):
    """Write PED/MAP files. Phenotype embedded in PED column 6."""
    n, p = X.shape
    with open(os.path.join(tmpdir, name + '.map'), 'w') as f:
        for i in range(p):
            f.write('1\tSNP_%d\t0\t%d\n' % (i, i + 1))
    with open(os.path.join(tmpdir, name + '.ped'), 'w') as f:
        for i in range(n):
            pheno = '%f' % float(Y[i]) if Y is not None else '-9'
            row = ['FAM%d' % i, 'IND%d' % i, '0', '0', '0', pheno]
            for j in range(p):
                a1, a2 = ALLELES[int(np.clip(round(float(X[i, j])), 0, 2))]
                row += [a1, a2]
            f.write(' '.join(row) + '\n')


def _run(cmd, tmpdir):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
    return r.returncode == 0, r.stdout + r.stderr


def _parse_assoc_linear(path):
    """Parse plink1.9 --linear output (.assoc.linear). Returns list of (snp_idx, p)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('CHR'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            snp = parts[1]   # SNP_i
            test = parts[4]  # ADD
            pval = parts[8]  # P
            if test != 'ADD':
                continue
            try:
                idx = int(snp.replace('SNP_', ''))
                p = float(pval) if pval != 'NA' else 1.0
                rows.append((idx, p))
            except:
                pass
    return rows


def _parse_epi_qt(path):
    """
    Parse plink1.9 .epi.qt file.
    Columns: CHR1 SNP1 CHR2 SNP2 BETA_INT STAT P
    Returns list of dicts with snp1, snp2, stat, p.
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('CHR'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                snp1 = int(parts[1].replace('SNP_', ''))
                snp2 = int(parts[3].replace('SNP_', ''))
                stat = float(parts[5]) if parts[5] != 'NA' else 0.0
                p    = float(parts[6]) if parts[6] != 'NA' else 1.0
                rows.append({'snp1': min(snp1, snp2),
                             'snp2': max(snp1, snp2),
                             'stat': stat, 'p': p})
            except:
                pass
    return rows


# ------------------------------------------------------------------ main pipeline

def plink2EpistasisPredict(X_train, Y_train, X_test, n_top_snps=None, n_top_pairs=None):
    """
    Run PLINK 1.9 epistasis pipeline and return (y_pred_test, top_pairs).

    Parameters
    ----------
    X_train, Y_train : training data
    X_test           : test data
    n_top_snps       : number of top marginal SNPs to restrict epistasis to
                       (default: max(10, min(2*sqrt(n_feats), 50)))
    n_top_pairs      : number of top interacting pairs to use as features

    Returns
    -------
    y_pred    : np.ndarray, predictions on X_test
    top_pairs : list of dicts with keys snp1, snp2, stat, p
    """
    n_feats = X_train.shape[1]
    if n_top_snps is None:
        n_top_snps = max(10, min(int(2 * np.sqrt(n_feats)), 50))
    if n_top_pairs is None:
        n_top_pairs = max(1, n_top_snps // 2)

    tmpdir = tempfile.mkdtemp(prefix='plink19epi_')
    try:
        # ---- 1. Full PED/MAP with phenotype in column 6 ----
        _write_ped_map(X_train, Y_train, tmpdir, name='data')

        # ---- 2. Marginal --linear -> top k SNPs ----
        ok, log = _run([PLINK19, '--ped', 'data.ped', '--map', 'data.map',
                        '--linear', '--allow-no-sex', '--pheno-name', 'PHENO1',
                        '--out', 'marginal', '--silent'], tmpdir)
        # plink1.9 uses column 6 as phenotype by default with --ped
        # if --linear fails, try without --pheno-name
        if not ok or not os.path.exists(os.path.join(tmpdir, 'marginal.assoc.linear')):
            ok, log = _run([PLINK19, '--ped', 'data.ped', '--map', 'data.map',
                            '--linear', '--allow-no-sex',
                            '--out', 'marginal', '--silent'], tmpdir)

        marginal = _parse_assoc_linear(os.path.join(tmpdir, 'marginal.assoc.linear'))
        if not marginal:
            raise RuntimeError('marginal --linear produced no results. log: ' + log)

        marginal.sort(key=lambda x: x[1])  # sort by p-value
        top_snps = [idx for idx, p in marginal[:n_top_snps]]

        if len(top_snps) < 2:
            raise RuntimeError('fewer than 2 top SNPs found')

        # ---- 3. Write subset PED/MAP with only top-k SNPs ----
        X_sub = X_train[:, top_snps]
        # remap SNP indices to 0..k-1 in the subset file
        _write_ped_map(X_sub, Y_train, tmpdir, name='sub')

        # ---- 4. Run --epistasis on the subset ----
        ok, log = _run([PLINK19, '--ped', 'sub.ped', '--map', 'sub.map',
                        '--epistasis', '--allow-no-sex',
                        '--out', 'epi', '--silent'], tmpdir)

        epi_rows = _parse_epi_qt(os.path.join(tmpdir, 'epi.epi.qt'))
        if not epi_rows:
            raise RuntimeError('--epistasis produced no .epi.qt output. log: ' + log)

        # Sort by |STAT| descending (most significant interaction first)
        epi_rows.sort(key=lambda x: abs(x['stat']), reverse=True)

        # Remap local subset indices back to original column indices
        top_pairs = []
        for r in epi_rows[:n_top_pairs]:
            orig1 = top_snps[r['snp1']] if r['snp1'] < len(top_snps) else r['snp1']
            orig2 = top_snps[r['snp2']] if r['snp2'] < len(top_snps) else r['snp2']
            top_pairs.append({'snp1': orig1, 'snp2': orig2,
                              'stat': r['stat'], 'p': r['p']})

        # ---- 5. Build features and fit Ridge ----
        def make_features(X, top_snps, top_pairs):
            parts = [X[:, top_snps]]
            for pair in top_pairs:
                i, j = pair['snp1'], pair['snp2']
                if i < X.shape[1] and j < X.shape[1]:
                    parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
            return np.hstack(parts)

        X_tr = make_features(X_train, top_snps, top_pairs)
        X_te = make_features(X_test,  top_snps, top_pairs)

        model = Ridge()
        model.fit(X_tr, Y_train)
        y_pred = model.predict(X_te)

        return y_pred, top_pairs

    except Exception as e:
        print('PLINK1.9 epistasis error:', e)
        return np.zeros(len(X_test)), []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
