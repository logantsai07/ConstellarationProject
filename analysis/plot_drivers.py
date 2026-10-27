"""plot_drivers.py
Bar plots of what drives coil cost, from all_results_with_descriptors.csv.

  drivers_betas_i{idx}.png    standardized rank regression coefficients
                              (cost ~ all descriptors), one bar per problem
  multipleR_i{idx}.png        multiple correlation R per problem
  raw_vs_partial_i{idx}.png   raw vs partial Spearman for edge iota, axis iota,
                              mirror ratio, aspect ratio (partial = holding the
                              other three fixed)

Higher cost = harder. Feasible means f_B <= (1+tol)*fB_target.

    python plot_drivers.py --csv all_results_with_descriptors.csv --target-index 3
"""

import argparse
import re
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

# nice short labels + plot order
SHORT = {"C_sparse_dipole": "C sparse", "D_thin_dipole": "D thin",
         "F_low_max_force_dipole": "F force", "I_low_curvature_filament": "I curv",
         "K_low_max_force_filament": "K force"}
PCOLOR = {"C_sparse_dipole": "#9ecae1", "D_thin_dipole": "#4292c6",
          "F_low_max_force_dipole": "#08519c", "I_low_curvature_filament": "#fb6a4a",
          "K_low_max_force_filament": "#cb181d"}

DESC_ORDER = ["mirror_ratio", "aspect_ratio", "iota_edge",
              "max_elongation", "vacuum_well", "triangularity"]
DESC_LABEL = {"mirror_ratio": "mirror", "aspect_ratio": "aspect", "iota_edge": "edge iota",
              "iota_axis": "axis iota", "max_elongation": "elong",
              "vacuum_well": "vac well", "triangularity": "triang"}


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def _rank(a):
    return pd.Series(np.asarray(a, float)).rank().to_numpy()


def _spearman(x, y):
    m = np.isfinite(np.asarray(x, float)) & np.isfinite(np.asarray(y, float))
    if m.sum() < 5:
        return float("nan")
    return _pearson(_rank(np.asarray(x)[m]), _rank(np.asarray(y)[m]))


def partial_spearman(y, x, Z):
    y = np.asarray(y, float); x = np.asarray(x, float)
    Zc = [np.asarray(z, float) for z in Z]
    mask = np.isfinite(y) & np.isfinite(x)
    for z in Zc:
        mask &= np.isfinite(z)
    if mask.sum() < len(Zc) + 3:
        return float("nan")
    ry, rx = _rank(y[mask]), _rank(x[mask])
    A = np.column_stack([np.ones(mask.sum())] + [_rank(z[mask]) for z in Zc])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    return _pearson(rx - A @ bx, ry - A @ by)


def rank_regression(y, cols):
    y = np.asarray(y, float)
    cols = [np.asarray(c, float) for c in cols]
    mask = np.isfinite(y)
    for c in cols:
        mask &= np.isfinite(c)
    if mask.sum() < len(cols) + 5:
        return None
    def z(a):
        r = _rank(a); s = r.std()
        return (r - r.mean()) / s if s > 0 else r * 0.0
    zy = z(y[mask]); Z = np.column_stack([z(c[mask]) for c in cols])
    beta, *_ = np.linalg.lstsq(Z, zy, rcond=None)
    resid = zy - Z @ beta
    r2 = 1.0 - (resid**2).sum() / (zy**2).sum() if (zy**2).sum() > 0 else float("nan")
    return {"beta": beta, "r2": r2, "n": int(mask.sum())}


def pick(cols, *patterns):
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def detect_descriptors(df):
    cols = list(df.columns)
    d = {}
    d["iota_edge"]      = pick(cols, r"edge.*rotational_transform.*field_periods", r"edge.*rotational_transform")
    d["iota_axis"]      = pick(cols, r"axis.*rotational_transform.*field_periods", r"axis.*rotational_transform")
    d["aspect_ratio"]   = pick(cols, r"metrics\.aspect_ratio$", r"(^|\.)aspect_ratio$", r"aspect_ratio$")
    d["max_elongation"] = pick(cols, r"metrics\.max_elongation", r"max_elongation", r"elongation")
    d["mirror_ratio"]   = pick(cols, r"edge.*mirror_ratio", r"mirror_ratio", r"mirror")
    d["vacuum_well"]    = pick(cols, r"vacuum_well")
    d["triangularity"]  = pick(cols, r"triangularity")
    return {k: v for k, v in d.items() if v}


def feas_cost_and_desc(df, problem, idx, desc, tol, normalize):
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
    for name, col in desc.items():
        v = pd.to_numeric(sub[col], errors="coerce").to_numpy()
        if "iota" in name:
            v = np.abs(v)
        vals[name] = v
    return cost, vals


# ------------------------------------------------------------------ figures
def fig_betas(df, idx, desc, tol, normalize, out):
    names = [k for k in DESC_ORDER if k in desc]
    probs = list(PROBLEMS)
    betas = {}      # problem -> array over names
    r2 = {}
    for p in probs:
        cost, vals = feas_cost_and_desc(df, p, idx, desc, tol, normalize)
        res = rank_regression(cost, [vals[k] for k in names])
        if res is None:
            betas[p] = np.full(len(names), np.nan); r2[p] = np.nan
        else:
            betas[p] = res["beta"]; r2[p] = res["r2"]

    x = np.arange(len(names)); w = 0.16
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 2, 5.5))
    for k, p in enumerate(probs):
        ax.bar(x + (k - 2) * w, betas[p], w, label=SHORT[p], color=PCOLOR[p])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([DESC_LABEL[n] for n in names])
    ax.set_ylabel("standardized regression coefficient")
    ax.set_title(f"What drives coil cost (i={idx}): cost ~ geometry, standardized betas\n"
                 "positive = raises cost (harder); dipoles blue, filaments red")
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  -> {out}")
    return r2


def fig_multipleR(r2, idx, out):
    probs = list(PROBLEMS)
    R = [np.sqrt(max(r2[p], 0)) if np.isfinite(r2[p]) else 0 for p in probs]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar([SHORT[p] for p in probs], R, color=[PCOLOR[p] for p in probs])
    for i, v in enumerate(R):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1); ax.set_ylabel("multiple correlation R")
    ax.set_title(f"How well geometry predicts coil cost (i={idx})\n"
                 "R = sqrt(R^2) from cost ~ all descriptors")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  -> {out}")


def fig_raw_vs_partial(df, idx, desc, tol, normalize, out):
    four = [k for k in ["iota_edge", "iota_axis", "mirror_ratio", "aspect_ratio"] if k in desc]
    probs = list(PROBLEMS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), squeeze=False)
    for ax, p in zip(axes.flat, probs):
        cost, vals = feas_cost_and_desc(df, p, idx, desc, tol, normalize)
        raw, par = [], []
        for d in four:
            raw.append(_spearman(vals[d], cost))
            controls = [vals[o] for o in four if o != d]
            par.append(partial_spearman(cost, vals[d], controls))
        xx = np.arange(len(four)); w = 0.38
        ax.bar(xx - w/2, raw, w, label="raw", color="#bdbdbd")
        ax.bar(xx + w/2, par, w, label="partial", color=("#2171b5" if p in DIPOLE else "#cb181d"))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xx); ax.set_xticklabels([DESC_LABEL[d] for d in four], fontsize=9)
        ax.set_ylim(-0.85, 0.85)
        ax.set_title(f"{SHORT[p]}  [{'dipole' if p in DIPOLE else 'filament'}]", fontsize=10)
        ax.set_ylabel("Spearman rho", fontsize=9); ax.legend(fontsize=8)
    for ax in axes.flat[len(probs):]:
        ax.axis("off")
    fig.suptitle(f"Raw vs partial correlation (i={idx}): partial holds the other three fixed.\n"
                 "edge iota collapses to ~0; mirror ratio survives", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def run(df, idx, desc, tol, normalize):
    print(f"\n=== budget i={idx} ===")
    r2 = fig_betas(df, idx, desc, tol, normalize, f"drivers_betas_i{idx}.png")
    fig_multipleR(r2, idx, f"multipleR_i{idx}.png")
    fig_raw_vs_partial(df, idx, desc, tol, normalize, f"raw_vs_partial_i{idx}.png")
    print("  multiple R per problem:  " +
          "  ".join(f"{SHORT[p]}={np.sqrt(max(r2[p],0)):.2f}" for p in PROBLEMS if np.isfinite(r2[p])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, low_memory=False)
    desc = detect_descriptors(df)
    print("descriptors detected: " + ", ".join(f"{k}" for k in desc))
    if a.summary:
        idxs = sorted(int(i) for i in pd.to_numeric(df["target_index"], errors="coerce").dropna().unique())
        for i in idxs:
            run(df, i, desc, a.tol, a.normalize)
    else:
        run(df, a.target_index, desc, a.tol, a.normalize)


if __name__ == "__main__":
    main()