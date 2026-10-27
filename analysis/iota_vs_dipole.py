"""iota_vs_dipole.py
Does high iota make coils harder, and is it worse for dipoles than filaments?
(checking the "high iota devices will be hard to make with dipoles" idea)

Input is all_results_with_descriptors.csv. At one f_B budget (feasible only):
  - scatter of iota vs achieved cost, one panel per problem
  - Spearman rho of cost vs iota for each problem, to compare dipoles (C, D, F)
    with filaments (I, K)
  - rho vs the other descriptors too (aspect ratio, elongation, mirror ratio,
    vacuum well)

Feasible means f_B <= (1 + tol) * fB_target, tol=0.01.

    python iota_vs_dipole.py --csv all_results_with_descriptors.csv --target-index 3
    python iota_vs_dipole.py --csv all_results_with_descriptors.csv --summary
    python iota_vs_dipole.py --csv all_results_with_descriptors.csv --normalize
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# problem -> achieved-objective column, grouped so we can compare families
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


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def _spearman(x, y):
    rx = pd.Series(np.asarray(x, float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, float)).rank().to_numpy()
    return _pearson(rx, ry)


def pick(cols, *patterns):
    """First column whose name matches any regex pattern (case-insensitive)."""
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def detect_descriptors(df):
    """Map friendly names -> actual column names present in the merged CSV."""
    cols = list(df.columns)
    d = {}
    # iota per field period (edge is the standard 'device iota' proxy for QI set)
    d["iota_edge_overN"] = pick(cols, r"edge.*rotational_transform.*field_periods",
                                      r"rotational_transform.*field_periods.*edge",
                                      r"edge.*rotational_transform")
    d["iota_axis_overN"] = pick(cols, r"axis.*rotational_transform.*field_periods",
                                      r"axis.*rotational_transform")
    d["aspect_ratio"]    = pick(cols, r"(^|\.)aspect_ratio$", r"aspect_ratio$", r"aspect")
    d["max_elongation"]  = pick(cols, r"max_elongation", r"elongation")
    d["mirror_ratio"]    = pick(cols, r"edge.*mirror_ratio", r"mirror_ratio", r"mirror")
    d["vacuum_well"]     = pick(cols, r"vacuum_well")
    d["triangularity"]   = pick(cols, r"triangularity")
    return {k: v for k, v in d.items() if v}


def nfp_of(df):
    # anchor so we don't match '...rotational_transform_over_n_field_periods'
    col = pick(list(df.columns), r"^nfp$", r"(^|\.)n_field_periods$")
    if col:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            return float(s.mode().iloc[0])
    return 3.0  # these runs are all nfp=3


def feasible_rows(df, problem, idx, tol):
    obj = PROBLEMS[problem]
    sub = df[(df["problem"] == problem) & (df["target_index"] == idx)].copy()
    for c in ("f_B", "fB_target", obj):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub[sub["f_B"] <= (1.0 + tol) * sub["fB_target"]]
    return sub


def cost_series(sub, problem, normalize):
    obj = PROBLEMS[problem]
    val = pd.to_numeric(sub[obj], errors="coerce")
    if normalize and "objective_unit" in sub.columns:
        val = val / pd.to_numeric(sub["objective_unit"], errors="coerce")
    return val


def correlation_table(df, idx, tol, normalize, desc, nfp):
    rows = []
    for fam, probs in (("dipole", DIPOLE), ("filament", FILAMENT)):
        for p in probs:
            sub = feasible_rows(df, p, idx, tol)
            if len(sub) < 5:
                rows.append({"family": fam, "problem": p, "n": len(sub)})
                continue
            cost = cost_series(sub, p, normalize)
            rec = {"family": fam, "problem": p, "n": int(cost.notna().sum())}
            for name, col in desc.items():
                x = pd.to_numeric(sub[col], errors="coerce")
                xv = x.abs() if "iota" in name else x   # "high iota" = large magnitude
                ok = xv.notna() & cost.notna()
                rec[f"rho_{name}"] = _spearman(xv[ok], cost[ok]) if ok.sum() >= 5 else float("nan")
            rows.append(rec)
    return pd.DataFrame(rows)


def scatter_grid(df, idx, tol, normalize, desc, nfp, outpath):
    iota_col = desc.get("iota_edge_overN") or desc.get("iota_axis_overN")
    if not iota_col:
        print("  no rotational-transform column found; skipping iota scatter.")
        return
    probs = list(PROBLEMS)
    ncol = 3
    nrow = int(np.ceil(len(probs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.3 * nrow), squeeze=False)

    for ax, p in zip(axes.flat, probs):
        sub = feasible_rows(df, p, idx, tol)
        cost = cost_series(sub, p, normalize)
        iota = pd.to_numeric(sub[iota_col], errors="coerce").abs() * nfp   # actual |iota|
        ok = iota.notna() & cost.notna() & (cost > 0)
        x, y = iota[ok].to_numpy(), cost[ok].to_numpy()
        fam = "dipole" if p in DIPOLE else "filament"
        color = "#1f77b4" if fam == "dipole" else "#d62728"
        ax.scatter(x, y, s=18, alpha=0.6, edgecolor="none", color=color)
        if len(y) >= 2 and (y > 0).all():
            ax.set_yscale("log")
        rho = _spearman(x, y) if len(x) >= 5 else float("nan")
        ax.set_title(f"{p}  [{fam}]\nn={len(x)}   Spearman rho={rho:+.2f}", fontsize=10)
        ax.set_xlabel(f"|iota| (edge, x{nfp:.0f})")
        tag = "cost / nescoil baseline" if normalize else PROBLEMS[p] + " (achieved)"
        ax.set_ylabel(tag, fontsize=9)

    for ax in axes.flat[len(probs):]:
        ax.axis("off")

    fig.suptitle(f"Rotational transform vs achieved coil cost   "
                 f"(target index i={idx}, f_B within {tol:.0%})",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def run_one(df, idx, tol, normalize, desc, nfp):
    print(f"\n==== target index i={idx}  (f_B within {tol:.0%}, "
          f"{'normalized' if normalize else 'raw achieved'} cost) ====")
    tbl = correlation_table(df, idx, tol, normalize, desc, nfp)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    out_csv = f"descriptor_correlations_i{idx}.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")
    scatter_grid(df, idx, tol, normalize, desc, nfp, f"iota_vs_cost_i{idx}.png")


def main():
    ap = argparse.ArgumentParser(description="iota (and other descriptors) vs achieved coil cost")
    ap.add_argument("--csv", required=True, help="all_results_with_descriptors.csv")
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--normalize", action="store_true",
                    help="divide cost by its nescoil baseline (objective_unit) before correlating")
    ap.add_argument("--summary", action="store_true",
                    help="run every target index present, not just one")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    desc = detect_descriptors(df)
    nfp = nfp_of(df)

    print("descriptor columns detected:")
    if not desc:
        print("  (none - did join_descriptors.py run? check the column names in the CSV)")
        return
    for k, v in desc.items():
        print(f"   {k:16s} <- {v}")
    print(f"nfp used for actual-iota scaling: {nfp:.0f}")

    if a.summary:
        idxs = sorted(int(i) for i in pd.to_numeric(df["target_index"], errors="coerce").dropna().unique())
        for i in idxs:
            run_one(df, i, a.tol, a.normalize, desc, nfp)
    else:
        run_one(df, a.target_index, a.tol, a.normalize, desc, nfp)


if __name__ == "__main__":
    main()