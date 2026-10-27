"""variance_explained.py
Which equilibrium parameter explains the most variation in coil cost?

The descriptors are correlated so univariate R^2 double counts. This uses the
Shapley/LMG decomposition instead (average incremental R^2 over all orderings
of the predictors), so the shares add up to the total R^2. Done on ranks since
cost spans orders of magnitude.

  variance_stacked_i{idx}.png   stacked bar per problem, gray = unexplained
  variance_grouped_i{idx}.png   same numbers grouped by parameter

    python variance_explained.py --csv all_results_with_descriptors.csv --target-index 3
"""

import argparse
import re
from math import factorial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIPOLE = {
    "C_sparse_dipole":        "f_l1_Phi",
    "D_thin_dipole":          "f_max_Phi",
    "F_low_max_force_dipole": "f_max_force_cyl",
}
FILAMENT = {
    "I_low_curvature_filament": "f_max_K_dot_grad_K_cyl",
    "K_low_max_force_filament": "f_max_force_cyl",
}
PROBLEMS = {**DIPOLE, **FILAMENT}
SHORT = {"C_sparse_dipole": "C sparse", "D_thin_dipole": "D thin",
         "F_low_max_force_dipole": "F force", "I_low_curvature_filament": "I curv",
         "K_low_max_force_filament": "K force"}

# predictors used in the decomposition; the two iotas are merged into one "iota"
# band in the plots but kept separate in the fit so collinearity is handled.
PREDS = ["iota_edge", "iota_axis", "mirror_ratio", "aspect_ratio",
         "max_elongation", "vacuum_well", "triangularity"]
# how predictors map onto plotted categories
CATEGORY = {"iota_edge": "iota", "iota_axis": "iota", "mirror_ratio": "mirror",
            "aspect_ratio": "aspect", "max_elongation": "elong",
            "vacuum_well": "vac well", "triangularity": "triang"}
CAT_ORDER = ["mirror", "aspect", "iota", "elong", "vac well", "triang"]
CAT_COLOR = {"mirror": "#08519c", "aspect": "#6baed6", "iota": "#f16913",
             "elong": "#41ab5d", "vac well": "#807dba", "triang": "#c994c7",
             "unexplained": "#d9d9d9"}


def _rank(a):
    return pd.Series(np.asarray(a, float)).rank().to_numpy()


def _zrank(a):
    r = _rank(a); s = r.std()
    return (r - r.mean()) / s if s > 0 else r * 0.0


def pick(cols, *patterns):
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def detect(df):
    cols = list(df.columns)
    d = {}
    d["iota_edge"]      = pick(cols, r"edge.*rotational_transform.*field_periods", r"edge.*rotational_transform")
    d["iota_axis"]      = pick(cols, r"axis.*rotational_transform.*field_periods", r"axis.*rotational_transform")
    d["aspect_ratio"]   = pick(cols, r"metrics\.aspect_ratio$", r"(^|\.)aspect_ratio$")
    d["max_elongation"] = pick(cols, r"metrics\.max_elongation", r"max_elongation", r"elongation")
    d["mirror_ratio"]   = pick(cols, r"edge.*mirror_ratio", r"mirror_ratio", r"mirror")
    d["vacuum_well"]    = pick(cols, r"vacuum_well")
    d["triangularity"]  = pick(cols, r"triangularity")
    return {k: v for k, v in d.items() if v}


def feas(df, problem, idx, desc, tol, normalize):
    obj = PROBLEMS[problem]
    sub = df[(df["problem"] == problem) & (df["target_index"] == idx)].copy()
    for c in ("f_B", "fB_target", obj):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub[np.isfinite(sub["f_B"]) & np.isfinite(sub["fB_target"])]
    sub = sub[sub["f_B"] <= (1.0 + tol) * sub["fB_target"]]
    cost = pd.to_numeric(sub[obj], errors="coerce")
    if normalize and "objective_unit" in sub.columns:
        cost = cost / pd.to_numeric(sub["objective_unit"], errors="coerce")
    cost = cost.to_numpy()
    vals = {}
    for name in PREDS:
        if desc.get(name):
            v = pd.to_numeric(sub[desc[name]], errors="coerce").to_numpy()
            if "iota" in name:
                v = np.abs(v)
            vals[name] = v
    return cost, vals


def _r2(zy, Z):
    if Z.shape[1] == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(Z, zy, rcond=None)
    resid = zy - Z @ beta
    sst = (zy**2).sum()
    return 1.0 - (resid**2).sum() / sst if sst > 0 else 0.0


def shapley_r2(cost, vals, preds):
    """Decomposition of rank-R^2. Returns (phi, uniq, total_R2, n) where
    phi[pred]  = Shapley/LMG share of R^2 (includes shared variance)
    uniq[pred] = semipartial R^2 (variance only this predictor explains)."""
    mask = np.isfinite(cost)
    for p in preds:
        mask &= np.isfinite(vals[p])
    n = int(mask.sum())
    if n < len(preds) + 5:
        return None, None, float("nan"), n
    zy = _zrank(cost[mask])
    Z = {p: _zrank(vals[p][mask]) for p in preds}
    k = len(preds)
    # precompute R^2 for every subset (bitmask)
    v = np.zeros(1 << k)
    for mbits in range(1 << k):
        cols = [Z[preds[j]] for j in range(k) if mbits & (1 << j)]
        v[mbits] = _r2(zy, np.column_stack(cols)) if cols else 0.0
    full = (1 << k) - 1
    phi, uniq = {}, {}
    for j in range(k):
        # Shapley value
        s = 0.0
        for mbits in range(1 << k):
            if mbits & (1 << j):
                continue
            ssz = bin(mbits).count("1")
            w = factorial(ssz) * factorial(k - ssz - 1) / factorial(k)
            s += w * (v[mbits | (1 << j)] - v[mbits])
        phi[preds[j]] = max(s, 0.0)                       # LMG is non-negative
        uniq[preds[j]] = max(v[full] - v[full ^ (1 << j)], 0.0)   # semipartial R^2
    return phi, uniq, float(v[full]), n


def to_categories(phi):
    cat = {c: 0.0 for c in CAT_ORDER}
    for p, val in phi.items():
        cat[CATEGORY[p]] += val
    return cat


def run(df, idx, desc, tol, normalize):
    preds = [p for p in PREDS if desc.get(p)]
    print(f"\n=== budget i={idx}  (variance explained in rank of coil cost) ===")
    cats_by_p = {}
    r2_by_p = {}
    for p in PROBLEMS:
        cost, vals = feas(df, p, idx, desc, tol, normalize)
        phi, uniq, R2, n = shapley_r2(cost, vals, preds)
        if phi is None:
            print(f"  {SHORT[p]:9s} n={n} (too few)")
            continue
        cat = to_categories(phi)
        ucat = to_categories(uniq)
        cats_by_p[p] = cat
        r2_by_p[p] = R2
        parts = "  ".join(f"{c}={cat[c]*100:4.1f}%" for c in CAT_ORDER)
        print(f"  {SHORT[p]:9s} n={n}  R2={R2:.2f}")
        print(f"      Shapley share : {parts}")
        print(f"      unique (only) : " +
              "  ".join(f"{c}={ucat[c]*100:4.1f}%" for c in CAT_ORDER) +
              f"   [iota unique: edge {uniq['iota_edge']*100:.1f}%, axis {uniq['iota_axis']*100:.1f}%]")

    _stacked(cats_by_p, r2_by_p, idx, f"variance_stacked_i{idx}.png")
    _grouped(cats_by_p, idx, f"variance_grouped_i{idx}.png")


def _stacked(cats_by_p, r2_by_p, idx, out):
    probs = [p for p in PROBLEMS if p in cats_by_p]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(probs))
    bottoms = np.zeros(len(probs))
    for c in CAT_ORDER:
        heights = np.array([cats_by_p[p][c] for p in probs])
        ax.bar(x, heights, 0.62, bottom=bottoms, label=c, color=CAT_COLOR[c])
        bottoms += heights
    unexpl = np.array([max(1 - r2_by_p[p], 0) for p in probs])
    ax.bar(x, unexpl, 0.62, bottom=bottoms, label="unexplained", color=CAT_COLOR["unexplained"])
    ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in probs])
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of coil-cost variance")
    ax.set_title(f"Variance in coil cost explained by each equilibrium parameter (i={idx})\n"
                 "colored = explained (Shapley share of R^2); grey = unexplained")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def _grouped(cats_by_p, idx, out):
    probs = [p for p in PROBLEMS if p in cats_by_p]
    x = np.arange(len(CAT_ORDER)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 5.2))
    pcol = {"C_sparse_dipole": "#9ecae1", "D_thin_dipole": "#4292c6",
            "F_low_max_force_dipole": "#08519c", "I_low_curvature_filament": "#fb6a4a",
            "K_low_max_force_filament": "#cb181d"}
    for k, p in enumerate(probs):
        heights = [cats_by_p[p][c] * 100 for c in CAT_ORDER]
        ax.bar(x + (k - 2) * w, heights, w, label=SHORT[p], color=pcol[p])
    ax.set_xticks(x); ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("% of coil-cost variance explained")
    ax.set_title(f"Variance explained per equilibrium parameter (i={idx})")
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False)
    desc = detect(df)
    print("descriptors detected: " + ", ".join(desc))
    if a.summary:
        idxs = sorted(int(i) for i in pd.to_numeric(df["target_index"], errors="coerce").dropna().unique())
        for i in idxs:
            run(df, i, desc, a.tol, a.normalize)
    else:
        run(df, a.target_index, desc, a.tol, a.normalize)


if __name__ == "__main__":
    main()