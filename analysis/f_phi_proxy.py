"""f_phi_proxy.py
Can f_Phi (fast L2 objective) stand in for sparse (f_l1_Phi, C) and thin
(f_max_Phi, D)? Needs all_results.csv from a --with-fphi run.

[1] ranking: Spearman of achieved f_Phi vs f_l1_Phi / f_max_Phi across configs
    feasible in both, at one f_B budget. high rho means configs that are hard
    for f_Phi are also hard for sparse/thin.

[2] solution quality: f_l1_Phi and f_max_Phi at the f_Phi optimum divided by
    what C and D get when they optimize those directly (always >= 1).
    close to 1 means the f_Phi coil is almost as sparse/thin.

    python f_phi_proxy.py --csv all_results.csv --target-index 3
    python f_phi_proxy.py --csv all_results.csv --summary
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# problem -> its achieved-objective column
FPHI = ("G_l2_dipole", "f_Phi")       # the fast L2 problem (added by --with-fphi)
SPARSE = ("C_sparse_dipole", "f_l1_Phi")
THIN = ("D_thin_dipole", "f_max_Phi")


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return _pearson(pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy())


def feasible(df, problem, idx, tol):
    sub = df[(df["problem"] == problem) & (df["target_index"] == idx)].copy()
    for c in ("f_B", "fB_target"):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub[np.isfinite(sub["f_B"]) & np.isfinite(sub["fB_target"])]
    return sub[sub["f_B"] <= (1.0 + tol) * sub["fB_target"]]


def col(df, name):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)


def analyse(df, idx, tol):
    gp, gobj = FPHI
    g = feasible(df, gp, idx, tol)
    if len(g) == 0:
        print(f"  i={idx}: no feasible G_l2_dipole rows (did --with-fphi run reach this budget?)")
        return None

    out = {"idx": idx}
    # assemble per-config frame for G with the metrics recorded at the f_Phi optimum
    G = pd.DataFrame({
        "config_id": g["config_id"].values,
        "fphi":        col(g, "f_Phi").values,
        "fphi_unit":   col(g, "objective_unit").values,
        "g_l1":        col(g, "f_l1_Phi").values,     # sparsity at the f_Phi solution
        "g_max":       col(g, "f_max_Phi").values,    # thinness at the f_Phi solution
    })

    rows = []
    for (cp, cobj), tag, gcol in ((SPARSE, "sparse", "g_l1"), (THIN, "thin", "g_max")):
        c = feasible(df, cp, idx, tol)
        C = pd.DataFrame({
            "config_id": c["config_id"].values,
            "cost":      col(c, cobj).values,
            "unit":      col(c, "objective_unit").values,
        })
        m = G.merge(C, on="config_id", how="inner").replace([np.inf, -np.inf], np.nan)
        m = m.dropna(subset=["fphi", "cost", gcol])
        n = len(m)
        if n < 3:
            rows.append({"vs": tag, "n": n}); continue

        # [1] ranking agreement (raw achieved, and normalised by each baseline)
        rho_raw = _spearman(m["fphi"], m["cost"])
        rho_norm = _spearman(m["fphi"] / m["fphi_unit"], m["cost"] / m["unit"])

        # [2] substitution quality: metric at the f_Phi solution / dedicated optimum
        ratio = (m[gcol] / m["cost"]).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[ratio > 0]
        rows.append({
            "vs": tag, "n": n,
            "rho_raw": rho_raw, "rho_norm": rho_norm,
            "ratio_med": float(ratio.median()),
            "ratio_p25": float(ratio.quantile(0.25)),
            "ratio_p75": float(ratio.quantile(0.75)),
            "within_25pct": float((ratio <= 1.25).mean()),
            "_ratio": ratio.values, "_m": m, "_gcol": gcol,
        })
    out["rows"] = rows
    return out


def report(res):
    idx = res["idx"]
    print(f"\n=== budget i={idx} ===")
    for r in res["rows"]:
        if r.get("rho_raw") is None or "rho_raw" not in r:
            print(f"  f_Phi vs {r['vs']:6s}: n={r.get('n',0)} (too few)"); continue
        print(f"  f_Phi vs {r['vs']:6s}:  n={r['n']:3d}   "
              f"rank agreement rho={r['rho_raw']:+.2f} (raw) / {r['rho_norm']:+.2f} (norm)   "
              f"substitution ratio median={r['ratio_med']:.2f} "
              f"[{r['ratio_p25']:.2f}-{r['ratio_p75']:.2f}], "
              f"{r['within_25pct']*100:.0f}% within 25%")


def plots(res, tol, outprefix):
    idx = res["idx"]
    rows = [r for r in res["rows"] if "rho_raw" in r]
    if not rows:
        return
    # scatter: f_Phi achieved vs sparse/thin achieved (normalised), log-log
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5.4), squeeze=False)
    for ax, r in zip(axes[0], rows):
        m = r["_m"]
        x = (m["fphi"] / m["fphi_unit"]).to_numpy()
        y = (m["cost"] / m["unit"]).to_numpy()
        ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=20, alpha=0.6, edgecolor="none", color="#1f77b4")
        if ok.sum() >= 2:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("f_Phi achieved / f_Phi baseline")
        ax.set_ylabel(f"{r['vs']} achieved / {r['vs']} baseline")
        ax.set_title(f"f_Phi vs {r['vs']}  (i={idx}, n={r['n']})\n"
                     f"Spearman rho={r['rho_norm']:+.2f} (norm)")
    fig.suptitle(f"Does f_Phi rank configs like sparse/thin? (f_B within {tol:.0%})", y=1.02)
    fig.tight_layout()
    p1 = f"{outprefix}_rank_i{idx}.png"; fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p1}")

    # histograms: substitution ratio (metric at f_Phi solution / dedicated optimum)
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.6), squeeze=False)
    for ax, r in zip(axes[0], rows):
        vals = r["_ratio"]
        ax.hist(vals, bins=np.linspace(1, max(2.0, np.quantile(vals, 0.95)), 25),
                color="#2ca02c", alpha=0.8)
        ax.axvline(1.0, color="k", lw=1, ls="--", label="as good as dedicated")
        ax.axvline(float(np.median(vals)), color="#d62728", lw=1.5,
                   label=f"median {np.median(vals):.2f}x")
        ax.set_xlabel(f"{r['vs']} at f_Phi solution / {r['vs']} optimum")
        ax.set_ylabel("configs"); ax.legend(fontsize=8)
        ax.set_title(f"Substitution cost, {r['vs']} (i={idx})")
    fig.suptitle("How much thin/sparse quality you give up by running f_Phi", y=1.02)
    fig.tight_layout()
    p2 = f"{outprefix}_ratio_i{idx}.png"; fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--outprefix", default="fphi_proxy")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, low_memory=False)
    if "G_l2_dipole" not in set(df["problem"].astype(str)):
        print("No G_l2_dipole rows in the CSV. Run quadcoil_study.py --with-fphi, then "
              "collect_results.py, before this."); return
    if "f_Phi" not in df.columns:
        print("No f_Phi column in the CSV. collect_results.py should add it from the "
              "G_l2_dipole out_dicts; re-run it on the output dir."); return

    idxs = ([int(i) for i in sorted(pd.to_numeric(df["target_index"], errors="coerce").dropna().unique())]
            if a.summary else [a.target_index])
    for i in idxs:
        res = analyse(df, i, a.tol)
        if res:
            report(res)
            plots(res, a.tol, a.outprefix)

    print("\nhigh rank rho: f_Phi is a good difficulty screen for that problem. "
          "ratio near 1: the f_Phi solution is about as good on that metric, so f_Phi "
          "could be run instead.")


if __name__ == "__main__":
    main()