"""iota_analysis.py
Checks whether high iota actually makes dipoles harder, or if it's just aspect
ratio. Input is all_results_with_descriptors.csv (from join_descriptors.py).

At one f_B budget, for each problem:
  [1] Spearman(cost, descriptor) with bootstrap 95% CI
  [2] partial Spearman of cost vs iota controlling for aspect ratio, mirror
      ratio, elongation (and same thing for mirror ratio)
  [3] rank regression of cost on all descriptors: betas, R^2, VIFs
  [4] feasibility vs |iota| terciles using every attempted config, since
      "hard to make" could also mean no feasible solution at all

Makes iota vs cost, mirror ratio vs cost, and feasibility by iota tercile plots.

Higher cost = harder. Feasible means f_B <= (1+tol)*fB_target, tol=0.01.

    python iota_analysis.py --csv all_results_with_descriptors.csv --target-index 3
    python iota_analysis.py --csv all_results_with_descriptors.csv --summary
    python iota_analysis.py --csv all_results_with_descriptors.csv --target-index 3 --boot 1000
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

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ stats utils
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
    return _pearson(_rank(x), _rank(y))


def _boot_ci(x, y, fn, nboot, lo=2.5, hi=97.5):
    """Percentile bootstrap CI for a correlation statistic fn(x,y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 8 or nboot <= 0:
        return (float("nan"), float("nan"))
    vals = np.empty(nboot)
    for b in range(nboot):
        idx = RNG.integers(0, n, n)
        vals[b] = fn(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))


def partial_spearman(y, x, Z):
    """Spearman partial correlation of y and x, controlling for columns in Z.

    Rank-transform everything, then correlate the residuals of y~Z and x~Z.
    Z is a list of 1-d arrays (the controls).
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    Zc = [np.asarray(z, float) for z in Z]
    mask = np.isfinite(y) & np.isfinite(x)
    for z in Zc:
        mask &= np.isfinite(z)
    if mask.sum() < len(Zc) + 3:
        return float("nan")
    ry, rx = _rank(y[mask]), _rank(x[mask])
    if not Zc:
        return _pearson(rx, ry)
    A = np.column_stack([np.ones(mask.sum())] + [_rank(z[mask]) for z in Zc])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    return _pearson(rx - A @ bx, ry - A @ by)


def rank_regression(y, X, names):
    """Standardized rank regression: z-scored ranks. Returns betas, R^2, VIFs."""
    y = np.asarray(y, float)
    cols = [np.asarray(c, float) for c in X]
    mask = np.isfinite(y)
    for c in cols:
        mask &= np.isfinite(c)
    if mask.sum() < len(cols) + 5:
        return None
    def z(a):
        r = _rank(a); s = r.std()
        return (r - r.mean()) / s if s > 0 else r * 0.0
    zy = z(y[mask])
    Z = np.column_stack([z(c[mask]) for c in cols])
    beta, *_ = np.linalg.lstsq(Z, zy, rcond=None)      # no intercept: all centered
    resid = zy - Z @ beta
    sstot = (zy**2).sum()
    r2 = 1.0 - (resid**2).sum() / sstot if sstot > 0 else float("nan")
    # VIFs
    vifs = []
    for j in range(Z.shape[1]):
        others = np.delete(Z, j, axis=1)
        if others.shape[1] == 0:
            vifs.append(1.0); continue
        bj, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        rj = Z[:, j] - others @ bj
        sst = (Z[:, j]**2).sum()
        r2j = 1.0 - (rj**2).sum() / sst if sst > 0 else 0.0
        vifs.append(1.0 / (1.0 - r2j) if r2j < 0.999 else float("inf"))
    return {"names": names, "beta": beta, "r2": r2, "vif": vifs, "n": int(mask.sum())}


# ------------------------------------------------------------------ data access
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


def nfp_of(df):
    col = pick(list(df.columns), r"^nfp$", r"(^|\.)n_field_periods$")
    if col:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            return float(s.mode().iloc[0])
    return 3.0


def attempted(df, problem, idx):
    """All attempted rows for a problem at a budget (feasible or not)."""
    obj = PROBLEMS[problem]
    sub = df[(df["problem"] == problem) & (df["target_index"] == idx)].copy()
    for c in ("f_B", "fB_target", obj):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub[np.isfinite(sub["f_B"]) & np.isfinite(sub["fB_target"])]
    sub["feasible"] = (sub["f_B"] <= (1.0 + ATT_TOL) * sub["fB_target"]).astype(float)
    return sub


def feas_subset(sub, problem, normalize):
    obj = PROBLEMS[problem]
    f = sub[sub["feasible"] > 0].copy()
    cost = pd.to_numeric(f[obj], errors="coerce")
    if normalize and "objective_unit" in f.columns:
        cost = cost / pd.to_numeric(f["objective_unit"], errors="coerce")
    f["_cost"] = cost
    return f


def descr_vals(frame, desc, name):
    col = desc.get(name)
    if col is None or col not in frame.columns:
        return None
    v = pd.to_numeric(frame[col], errors="coerce").to_numpy()
    if "iota" in name:
        v = np.abs(v)          # "high iota" = large magnitude
    return v


# ------------------------------------------------------------------ report
def run_budget(df, idx, desc, nfp, normalize, nboot):
    print(f"\n{'='*74}\n budget i={idx}   (f_B within {ATT_TOL:.0%};  "
          f"cost = {'normalized' if normalize else 'raw achieved'}, higher = harder)\n{'='*74}")

    order = ["iota_edge", "iota_axis", "mirror_ratio", "aspect_ratio",
             "max_elongation", "vacuum_well", "triangularity"]
    order = [k for k in order if k in desc]

    # counts
    counts = {p: int((attempted(df, p, idx)["feasible"] > 0).sum()) for p in PROBLEMS}
    print("feasible n:  " + "   ".join(f"{p.split('_')[0]}={counts[p]}" for p in PROBLEMS))

    # [1] univariate with CI  +  [2] partial correlations
    controls_for = {
        "iota_edge":    ["aspect_ratio", "mirror_ratio", "max_elongation"],
        "iota_axis":    ["aspect_ratio", "mirror_ratio", "max_elongation"],
        "mirror_ratio": ["aspect_ratio", "iota_edge", "max_elongation"],
    }
    for fam, probs in (("DIPOLE", DIPOLE), ("FILAMENT", FILAMENT)):
        for p in probs:
            sub = attempted(df, p, idx)
            f = feas_subset(sub, p, normalize)
            cost = f["_cost"].to_numpy()
            n = np.isfinite(cost).sum()
            print(f"\n  {p}  [{fam.lower()}]  n={n}")
            if n < 8:
                print("    too few feasible to correlate.")
                continue
            for name in order:
                v = descr_vals(f, desc, name)
                if v is None:
                    continue
                m = np.isfinite(v) & np.isfinite(cost)
                if m.sum() < 8:
                    continue
                rho = _spearman(v[m], cost[m])
                lo, hi = _boot_ci(v[m], cost[m], _spearman, nboot)
                star = "" if (np.isnan(lo) or np.isnan(hi) or (lo < 0 < hi)) else "  *"
                ci = f"[{lo:+.2f},{hi:+.2f}]" if not np.isnan(lo) else "[  n/a  ]"
                extra = ""
                if name in controls_for:
                    ctrls = [descr_vals(f, desc, c) for c in controls_for[name] if desc.get(c)]
                    ctrls = [c for c in ctrls if c is not None]
                    pr = partial_spearman(cost, v, ctrls)
                    extra = f"   partial={pr:+.2f} (hold {'+'.join(controls_for[name])})"
                print(f"     {name:14s} rho={rho:+.2f} {ci}{star}{extra}")

    # [3] rank regression per problem
    print(f"\n  --- rank multiple regression: cost ~ descriptors (standardized betas) ---")
    reg_names = [k for k in ["iota_edge", "aspect_ratio", "mirror_ratio",
                             "max_elongation", "vacuum_well", "triangularity"] if k in desc]
    for p in PROBLEMS:
        sub = attempted(df, p, idx)
        f = feas_subset(sub, p, normalize)
        cost = f["_cost"].to_numpy()
        cols = [descr_vals(f, desc, k) for k in reg_names]
        res = rank_regression(cost, cols, reg_names)
        if res is None:
            print(f"    {p:26s}  (too few)")
            continue
        betas = "  ".join(f"{nm.split('_')[0]}:{b:+.2f}" for nm, b in zip(res["names"], res["beta"]))
        maxvif = max(v for v in res["vif"] if np.isfinite(v)) if res["vif"] else float("nan")
        print(f"    {p:26s} R2={res['r2']:.2f} n={res['n']}  {betas}   maxVIF={maxvif:.1f}")

    # [4] feasibility vs iota (all attempted)
    print(f"\n  --- feasibility vs |edge iota|: rate by tercile (all attempted configs) ---")
    ic = desc.get("iota_edge")
    if ic:
        for p in PROBLEMS:
            sub = attempted(df, p, idx)
            if len(sub) < 9:
                print(f"    {p:26s}  (only {len(sub)} attempted)")
                continue
            iota = np.abs(pd.to_numeric(sub[ic], errors="coerce").to_numpy())
            feas = sub["feasible"].to_numpy()
            m = np.isfinite(iota) & np.isfinite(feas)
            iota, feas = iota[m], feas[m]
            # terciles by iota
            try:
                qs = np.quantile(iota, [1/3, 2/3])
                lo_m = iota <= qs[0]; mid_m = (iota > qs[0]) & (iota <= qs[1]); hi_m = iota > qs[1]
                rates = [feas[g].mean() if g.sum() else float("nan") for g in (lo_m, mid_m, hi_m)]
            except Exception:
                rates = [float("nan")] * 3
            rho = _spearman(iota, feas) if feas.std() > 0 else float("nan")
            print(f"    {p:26s} low/mid/high iota feas% = "
                  f"{rates[0]*100:4.0f} /{rates[1]*100:4.0f} /{rates[2]*100:4.0f}   "
                  f"overall={feas.mean()*100:4.0f}%   rho(feas,iota)={rho:+.2f}")

    # scatters
    _scatter_grid(df, idx, desc, nfp, normalize, "iota_edge",
                  f"iota_vs_cost_i{idx}.png", xlabel=f"|iota| edge (x{nfp:.0f})", scale_nfp=True)
    _scatter_grid(df, idx, desc, nfp, normalize, "mirror_ratio",
                  f"mirror_vs_cost_i{idx}.png", xlabel="edge mirror ratio", scale_nfp=False)
    _feas_bar(df, idx, desc, f"feasibility_vs_iota_i{idx}.png")


def _scatter_grid(df, idx, desc, nfp, normalize, xname, outpath, xlabel, scale_nfp):
    xcol = desc.get(xname)
    if not xcol:
        return
    probs = list(PROBLEMS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), squeeze=False)
    for ax, p in zip(axes.flat, probs):
        sub = attempted(df, p, idx)
        f = feas_subset(sub, p, normalize)
        cost = f["_cost"].to_numpy()
        x = pd.to_numeric(f[xcol], errors="coerce").to_numpy()
        if "iota" in xname:
            x = np.abs(x)
        if scale_nfp:
            x = x * nfp
        ok = np.isfinite(x) & np.isfinite(cost) & (cost > 0)
        x, y = x[ok], cost[ok]
        fam = "dipole" if p in DIPOLE else "filament"
        ax.scatter(x, y, s=16, alpha=0.55, edgecolor="none",
                   color="#1f77b4" if fam == "dipole" else "#d62728")
        if len(y) >= 2 and (y > 0).all():
            ax.set_yscale("log")
        rho = _spearman(x, y) if len(x) >= 5 else float("nan")
        ax.set_title(f"{p} [{fam}]\nn={len(x)}  rho={rho:+.2f}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(("cost/baseline" if normalize else PROBLEMS[p]), fontsize=8)
    for ax in axes.flat[len(probs):]:
        ax.axis("off")
    fig.suptitle(f"{xlabel}  vs  achieved coil cost   (i={idx}, f_B within {ATT_TOL:.0%})", y=1.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def _feas_bar(df, idx, desc, outpath):
    ic = desc.get("iota_edge")
    if not ic:
        return
    probs = list(PROBLEMS)
    labels = ["low", "mid", "high"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.15
    xpos = np.arange(3)
    for k, p in enumerate(probs):
        sub = attempted(df, p, idx)
        if len(sub) < 9:
            continue
        iota = np.abs(pd.to_numeric(sub[ic], errors="coerce").to_numpy())
        feas = sub["feasible"].to_numpy()
        m = np.isfinite(iota) & np.isfinite(feas)
        iota, feas = iota[m], feas[m]
        qs = np.quantile(iota, [1/3, 2/3])
        groups = [iota <= qs[0], (iota > qs[0]) & (iota <= qs[1]), iota > qs[1]]
        rates = [feas[g].mean()*100 if g.sum() else 0 for g in groups]
        ax.bar(xpos + (k - 2) * width, rates, width,
               label=p.split("_")[0], color=("#1f77b4" if p in DIPOLE else "#d62728"),
               alpha=0.55 + 0.1 * k)
    ax.set_xticks(xpos); ax.set_xticklabels([f"{l} |iota|" for l in labels])
    ax.set_ylabel("feasible %"); ax.set_ylim(0, 105)
    ax.set_title(f"Dipole/filament feasibility by |edge iota| tercile (i={idx})")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)
    print(f"  -> {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--summary", action="store_true", help="run all budgets")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for CIs (0 to disable)")
    a = ap.parse_args()

    global ATT_TOL
    ATT_TOL = a.tol

    df = pd.read_csv(a.csv, low_memory=False)
    desc = detect_descriptors(df)
    nfp = nfp_of(df)
    print("descriptors detected:")
    for k, v in desc.items():
        print(f"   {k:14s} <- {v}")
    print(f"nfp for actual-iota scaling: {nfp:.0f}")
    if "iota_edge" not in desc:
        print("WARNING: no edge rotational transform column found; iota analysis limited.")

    if a.summary:
        idxs = sorted(int(i) for i in pd.to_numeric(df["target_index"], errors="coerce").dropna().unique())
        for i in idxs:
            run_budget(df, i, desc, nfp, a.normalize, a.boot)
    else:
        run_budget(df, a.target_index, desc, nfp, a.normalize, a.boot)


if __name__ == "__main__":
    main()