"""plot_tradeoffs.py
Scatter of each equilibrium's optimal objective in problem A vs problem B at
target index i, only using equilibria feasible in both.

Feasible means f_B <= (1 + tol) * fB_target (within 1%, tol=0.01).
A downward trend means what helps one coil metric hurts the other.

    # dipole sparsity vs filament curvature
    python plot_tradeoffs.py --csv all_results.csv \
        --problem-a C_sparse_dipole --problem-b I_low_curvature_filament \
        --target-index 2

    # correlation table for every pair and budget
    python plot_tradeoffs.py --csv all_results.csv --summary
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# problem -> the objective it minimises (its achieved value lives in this column)
PROBLEMS = {
    "C_sparse_dipole":          "f_l1_Phi",
    "D_thin_dipole":            "f_max_Phi",
    "F_low_max_force_dipole":   "f_max_force_cyl",
    "I_low_curvature_filament": "f_max_K_dot_grad_K_cyl",
    "K_low_max_force_filament": "f_max_force_cyl",
}


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def _spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return _pearson(rx, ry)


def load(csv):
    df = pd.read_csv(csv)
    numeric = {"f_B", "fB_target", "objective_unit", "target_index"} | set(PROBLEMS.values())
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def subset(df, problem, target_index, tol, normalize):
    """Feasible equilibria for one problem at one target index -> (config_id, val)."""
    obj = PROBLEMS[problem]
    sub = df[(df["problem"] == problem) & (df["target_index"] == target_index)].copy()
    sub = sub[sub["f_B"] <= (1.0 + tol) * sub["fB_target"]]        # within tol of target
    val = sub[obj].astype(float)
    if normalize and "objective_unit" in sub.columns:
        val = val / sub["objective_unit"].astype(float)
    out = pd.DataFrame({"config_id": sub["config_id"].values, "val": val.values})
    return out.replace([np.inf, -np.inf], np.nan).dropna(), obj


def scatter(df, pa, pb, idx, tol, normalize, outpath):
    A, objA = subset(df, pa, idx, tol, normalize)
    B, objB = subset(df, pb, idx, tol, normalize)
    m = A.merge(B, on="config_id", suffixes=("_a", "_b"))
    n = len(m)
    if n < 3:
        print(f"  {pa} vs {pb} @ i={idx}: only {n} equilibria feasible in both (need >=3).")
        return

    x, y = m["val_a"].to_numpy(), m["val_b"].to_numpy()
    pos = (x > 0) & (y > 0)
    sr = _spearman(x, y)
    pr = _pearson(np.log10(x[pos]), np.log10(y[pos])) if pos.sum() >= 2 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolor="none")
    if pos.all():
        ax.set_xscale("log"); ax.set_yscale("log")
    tag = " / nescoil baseline" if normalize else " (achieved)"
    ax.set_xlabel(f"{pa}\n{objA}{tag}")
    ax.set_ylabel(f"{pb}\n{objB}{tag}")
    ax.set_title(f"target index i={idx}   f_B within {tol:.0%} of target\n"
                 f"n={n} equilibria    Spearman rho={sr:+.2f}    Pearson(log)={pr:+.2f}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    pairs_csv = outpath.rsplit(".", 1)[0] + "_pairs.csv"
    m.rename(columns={"val_a": f"{pa}:{objA}", "val_b": f"{pb}:{objB}"}).to_csv(pairs_csv, index=False)
    print(f"  {pa} vs {pb} @ i={idx}:  n={n}  Spearman={sr:+.3f}  Pearson(log)={pr:+.3f}")
    print(f"    -> {outpath}")
    print(f"    -> {pairs_csv}")


def summary(df, tol, normalize):
    probs = list(PROBLEMS)
    idxs = sorted(int(i) for i in df["target_index"].dropna().unique())
    print(f"Spearman rho, feasible-in-both (f_B within {tol:.0%} of target)"
          f"{', normalised by nescoil baseline' if normalize else ''}:\n")
    for i in idxs:
        print(f"-- target index i={i} --")
        for a in range(len(probs)):
            for b in range(a + 1, len(probs)):
                A, _ = subset(df, probs[a], i, tol, normalize)
                B, _ = subset(df, probs[b], i, tol, normalize)
                m = A.merge(B, on="config_id", suffixes=("_a", "_b"))
                if len(m) < 5:
                    print(f"  {probs[a]:26s} vs {probs[b]:26s}  n={len(m):3d}  (too few)")
                else:
                    sr = _spearman(m["val_a"], m["val_b"])
                    print(f"  {probs[a]:26s} vs {probs[b]:26s}  n={len(m):3d}  rho={sr:+.2f}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Per-equilibrium trade-off scatter between two quadcoil problems")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--problem-a", default="C_sparse_dipole", choices=list(PROBLEMS))
    ap.add_argument("--problem-b", default="I_low_curvature_filament", choices=list(PROBLEMS))
    ap.add_argument("--target-index", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.01, help="feasibility tolerance on f_B (default 0.01 = 1%)")
    ap.add_argument("--normalize", action="store_true",
                    help="divide each objective by its own nescoil baseline (objective_unit)")
    ap.add_argument("--summary", action="store_true",
                    help="print Spearman for every problem pair at every target index")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = load(a.csv)
    if a.summary:
        summary(df, a.tol, a.normalize)
        return
    out = a.out or f"tradeoff_{a.problem_a}_vs_{a.problem_b}_i{a.target_index}.png"
    scatter(df, a.problem_a, a.problem_b, a.target_index, a.tol, a.normalize, out)


if __name__ == "__main__":
    main()