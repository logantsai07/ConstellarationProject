"""multiple_corr_metrics.py
Multiple correlation R between each problem's cost and all the metrics.*
columns together, i.e. how well the equilibrium metrics predict coil cost.

One R per problem per budget, done on ranks (Spearman) and on log(cost).
Also prints adjusted R^2 since there are ~12 predictors.

Input: all_results_with_descriptors.csv. Outputs a problem x budget table of R,
multipleR_allmetrics_i{idx}.png, and per_column_corr_i{idx}.csv.

    python multiple_corr_metrics.py --csv all_results_with_descriptors.csv --summary
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# problem -> achieved-objective column (its "cost")
PROBLEMS = {
    "C_sparse_dipole":          "f_l1_Phi",
    "D_thin_dipole":            "f_max_Phi",
    "F_low_max_force_dipole":   "f_max_force_cyl",
    "G_l2_dipole":              "f_Phi",
    "I_low_curvature_filament": "f_max_K_dot_grad_K_cyl",
    "K_low_max_force_filament": "f_max_force_cyl",
}
SHORT = {"C_sparse_dipole": "C sparse", "D_thin_dipole": "D thin",
         "F_low_max_force_dipole": "F force", "G_l2_dipole": "G f_Phi",
         "I_low_curvature_filament": "I curv", "K_low_max_force_filament": "K force"}
COLOR = {"C_sparse_dipole": "#9ecae1", "D_thin_dipole": "#4292c6",
         "F_low_max_force_dipole": "#08519c", "G_l2_dipole": "#6a51a3",
         "I_low_curvature_filament": "#fb6a4a", "K_low_max_force_filament": "#cb181d"}


def _rank(a):
    return pd.Series(np.asarray(a, float)).rank().to_numpy()


def _z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 0 else a * 0.0


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def metrics_columns(df):
    """Every metrics.json column: metrics.* numeric with nonzero variance."""
    cols = []
    for c in df.columns:
        if not c.startswith("metrics."):
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() > 10 and v.std(skipna=True) > 0:
            cols.append(c)
    return cols


def multiple_R(cost, Xcols, mode):
    """Coefficient of multiple correlation R (+ R^2, adjusted R^2, n)."""
    cost = np.asarray(cost, float)
    cols = [np.asarray(c, float) for c in Xcols]
    mask = np.isfinite(cost)
    if mode == "log":
        mask &= cost > 0
    for c in cols:
        mask &= np.isfinite(c)
    n = int(mask.sum())
    p = len(cols)
    if n < p + 3:
        return float("nan"), float("nan"), float("nan"), n
    if mode == "rank":
        y = _z(_rank(cost[mask]))
        Z = np.column_stack([_z(_rank(c[mask])) for c in cols])
    else:  # log
        y = _z(np.log(cost[mask]))
        Z = np.column_stack([_z(c[mask]) for c in cols])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    sst = (y**2).sum()
    r2 = 1.0 - (resid**2).sum() / sst if sst > 0 else float("nan")
    r2 = max(min(r2, 1.0), 0.0)
    r2adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
    return float(np.sqrt(r2)), r2, r2adj, n


def feasible(df, problem, idx, tol):
    obj = PROBLEMS[problem]
    if obj not in df.columns:
        return df.iloc[0:0]                      # objective column not in CSV yet
    sub = df[(df["problem"] == problem) & (df["target_index"] == idx)].copy()
    if len(sub) == 0:
        return sub
    for c in ("f_B", "fB_target", obj):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub[np.isfinite(sub["f_B"]) & np.isfinite(sub["fB_target"])]
    return sub[sub["f_B"] <= (1.0 + tol) * sub["fB_target"]]


def run_budget(df, idx, metrics, tol):
    print(f"\n=== budget i={idx}   (R = coefficient of multiple correlation, "
          f"cost ~ all {len(metrics)} metrics.* columns) ===")
    print(f"{'problem':16s} {'n':>5s}  {'R(rank)':>8s} {'R2adj(rk)':>9s}  {'R(log)':>7s}")
    results = {}
    percol = {}
    for p in PROBLEMS:
        sub = feasible(df, p, idx, tol)
        if len(sub) == 0:
            continue
        cost = pd.to_numeric(sub[PROBLEMS[p]], errors="coerce").to_numpy()
        Xcols = [pd.to_numeric(sub[m], errors="coerce").to_numpy() for m in metrics]
        R_r, r2_r, adj_r, n = multiple_R(cost, Xcols, "rank")
        R_l, r2_l, adj_l, _ = multiple_R(cost, Xcols, "log")
        if np.isnan(R_r):
            print(f"{SHORT[p]:16s} {n:5d}  (too few)")
            continue
        results[p] = R_r
        print(f"{SHORT[p]:16s} {n:5d}  {R_r:8.2f} {adj_r:9.2f}  {R_l:7.2f}")
        # univariate Spearman for each column
        percol[SHORT[p]] = {m.replace("metrics.", ""): _spearman(
            pd.to_numeric(sub[m], errors="coerce").to_numpy(), cost) for m in metrics}

    if percol:
        pc = pd.DataFrame(percol).T
        out = f"per_column_corr_i{idx}.csv"
        pc.to_csv(out)
        print(f"  per-column univariate Spearman -> {out}")
        _heatmap(percol, idx, metrics)

    if results:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ps = [p for p in PROBLEMS if p in results]
        ax.bar([SHORT[p] for p in ps], [results[p] for p in ps],
               color=[COLOR[p] for p in ps])
        for i, p in enumerate(ps):
            ax.text(i, results[p] + 0.01, f"{results[p]:.2f}", ha="center", fontsize=10)
        ax.set_ylim(0, 1); ax.set_ylabel("R (multiple correlation)")
        ax.set_title(f"How well all equilibrium metrics predict each coil cost (i={idx})\n"
                     f"R = sqrt(R^2), cost ~ all {len(metrics)} metrics.* columns, rank-based")
        fig.tight_layout()
        out = f"multipleR_allmetrics_i{idx}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  -> {out}")


def _heatmap(percol, idx, metrics):
    """Colormap of Spearman(cost, metric): rows = problems, cols = metrics."""
    probs = list(percol.keys())
    mcols = [m.replace("metrics.", "") for m in metrics]
    M = np.array([[percol[p].get(mc, np.nan) for mc in mcols] for p in probs], float)
    fig, ax = plt.subplots(figsize=(0.95 * len(mcols) + 3, 0.6 * len(probs) + 2.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mcols)))
    ax.set_xticklabels(mcols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(probs))); ax.set_yticklabels(probs)
    for i in range(len(probs)):
        for j in range(len(mcols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.5 else "black")
    ax.set_title(f"Spearman(coil cost, equilibrium metric), per problem (i={idx})\n"
                 "red = raises cost (harder), blue = lowers cost (easier)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
    fig.tight_layout()
    out = f"corr_heatmap_i{idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, low_memory=False)
    metrics = metrics_columns(df)
    if not metrics:
        print("No metrics.* columns found. Run join_descriptors.py on all_results.csv first "
              "so the metrics.json columns are merged in."); return
    print(f"{len(metrics)} metrics.json columns used as predictors:")
    for m in metrics:
        print("   ", m)

    idxs = ([int(i) for i in sorted(pd.to_numeric(df["target_index"], errors="coerce").dropna().unique())]
            if a.summary else [a.target_index])
    for i in idxs:
        run_budget(df, i, metrics, a.tol)


if __name__ == "__main__":
    main()