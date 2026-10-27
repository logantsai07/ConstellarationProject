"""pca_plasma.py
PCA on the plasma parameters, then PC regression of coil cost.

  1) how many independent directions do the parameters have (scree plot)
  2) how many PCs do you need to predict cost (cumulative R^2)

PCA only looks at spread in the parameters, not cost, so (2) is the one that
matters for the coils.

  pca_scree.png         explained variance per PC + cumulative
  pca_loadings.png      loadings of the first PCs
  pcr_cumR2_i{idx}.png  cumulative R^2 vs number of PCs, per problem

Everything is standardized.
    python pca_plasma.py --csv all_results_with_descriptors.csv --target-index 3
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIPOLE = {"C_sparse_dipole": "f_l1_Phi", "D_thin_dipole": "f_max_Phi",
          "F_low_max_force_dipole": "f_max_force_cyl"}
FILAMENT = {"I_low_curvature_filament": "f_max_K_dot_grad_K_cyl",
            "K_low_max_force_filament": "f_max_force_cyl"}
PROBLEMS = {**DIPOLE, **FILAMENT}
SHORT = {"C_sparse_dipole": "C sparse", "D_thin_dipole": "D thin",
         "F_low_max_force_dipole": "F force", "I_low_curvature_filament": "I curv",
         "K_low_max_force_filament": "K force"}
PCOLOR = {"C_sparse_dipole": "#9ecae1", "D_thin_dipole": "#4292c6",
          "F_low_max_force_dipole": "#08519c", "I_low_curvature_filament": "#fb6a4a",
          "K_low_max_force_filament": "#cb181d"}


def short_param(col):
    return (col.replace("metrics.", "")
               .replace("_over_n_field_periods", "/N")
               .replace("rotational_transform", "iota")
               .replace("magnetic_", "")
               .replace("minimum_normalized_", "min_")
               .replace("flux_compression_in_regions_of_bad_curvature", "flux_compression"))


def select_params(df):
    """Physics equilibrium metrics with real variance; drop derived ratios/constants."""
    cols = []
    for c in df.columns:
        if not c.startswith("metrics."):
            continue
        if "aspect_ratio_over" in c:        # derived ratio of two others; skip
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() > 10 and v.std(skipna=True) > 0:
            cols.append(c)
    return cols


def _rank(a):
    return pd.Series(np.asarray(a, float)).rank().to_numpy()


def _zrank(a):
    r = _rank(a); s = r.std()
    return (r - r.mean()) / s if s > 0 else r * 0.0


def _r2(y, X):
    if X.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    sst = ((y - y.mean())**2).sum()
    return 1.0 - (resid**2).sum() / sst if sst > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-index", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--normalize", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, low_memory=False)
    params = select_params(df)
    if len(params) < 3:
        print("too few plasma-parameter columns found:", params); return
    print(f"{len(params)} plasma parameters used for PCA:")
    for c in params:
        print("   ", short_param(c), " <-", c)

    # one row per config, complete cases
    cfg = df.drop_duplicates("config_id")[["config_id"] + params].copy()
    for c in params:
        cfg[c] = pd.to_numeric(cfg[c], errors="coerce")
    cfg = cfg.dropna(subset=params)
    X = cfg[params].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    print(f"\nPCA on {Xz.shape[0]} configs x {Xz.shape[1]} parameters")

    # PCA via SVD
    U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
    evr = s**2 / (s**2).sum()
    cum = np.cumsum(evr)
    # sign convention: largest-magnitude loading positive
    for i in range(Vt.shape[0]):
        if Vt[i][np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] *= -1; U[:, i] *= -1
    scores = U * s                       # config scores, n x k

    print("\nexplained variance by principal component:")
    for i in range(len(evr)):
        print(f"  PC{i+1}: {evr[i]*100:5.1f}%   cumulative {cum[i]*100:5.1f}%")
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  -> {n90} PCs reach 90% of plasma-parameter variance, {n95} PCs reach 95%")

    print("\nloadings of the first 3 PCs (what each combination is):")
    for i in range(min(3, Vt.shape[0])):
        order = np.argsort(-np.abs(Vt[i]))
        terms = "  ".join(f"{Vt[i][j]:+.2f}*{short_param(params[j])}" for j in order[:4])
        print(f"  PC{i+1} ({evr[i]*100:.0f}%): {terms} ...")

    _scree(evr, cum, n90, n95, "pca_scree.png")
    _loadings(Vt, evr, params, "pca_loadings.png")

    # ---- PCR: cumulative R^2 vs number of PCs, per problem ----
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    scores_df.insert(0, "config_id", cfg["config_id"].values)

    print(f"\n--- PCR at i={a.target_index}: cumulative R^2(rank cost ~ first m PCs) ---")
    cum_by_p = {}
    for p in PROBLEMS:
        obj = PROBLEMS[p]
        sub = df[(df["problem"] == p) & (df["target_index"] == a.target_index)].copy()
        for c in ("f_B", "fB_target", obj):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub[np.isfinite(sub["f_B"]) & np.isfinite(sub["fB_target"])]
        sub = sub[sub["f_B"] <= (1.0 + a.tol) * sub["fB_target"]]
        cost = pd.to_numeric(sub[obj], errors="coerce")
        if a.normalize and "objective_unit" in sub.columns:
            cost = cost / pd.to_numeric(sub["objective_unit"], errors="coerce")
        m = pd.DataFrame({"config_id": sub["config_id"].values, "cost": cost.values}).dropna()
        m = m.merge(scores_df, on="config_id", how="inner")
        if len(m) < len(params) + 5:
            print(f"  {SHORT[p]:9s} n={len(m)} (too few)"); continue
        y = _zrank(m["cost"].to_numpy())
        S = m[[f"PC{i+1}" for i in range(scores.shape[1])]].to_numpy()
        S = (S - S.mean(0)) / np.where(S.std(0) == 0, 1, S.std(0))
        cumr2 = [_r2(y, S[:, :mm]) for mm in range(1, S.shape[1] + 1)]
        cum_by_p[p] = cumr2
        # which single PC is most predictive on its own
        singles = [abs(np.corrcoef(_zrank(m["cost"]), S[:, i])[0, 1]) for i in range(S.shape[1])]
        best = int(np.argmax(singles)) + 1
        print(f"  {SHORT[p]:9s} n={len(m)}  full R2={cumr2[-1]:.2f}  "
              f"R2 with 1 PC={cumr2[0]:.2f}, 2={cumr2[1]:.2f}, 3={cumr2[2]:.2f}  "
              f"(most predictive single: PC{best})")

    _pcr(cum_by_p, a.target_index, f"pcr_cumR2_i{a.target_index}.png")


def _scree(evr, cum, n90, n95, out):
    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, evr * 100, color="#6baed6", label="per PC")
    ax.plot(x, cum * 100, "-o", color="#08519c", label="cumulative")
    ax.axhline(90, ls="--", color="grey", lw=0.8)
    ax.axvline(n90, ls=":", color="k", lw=0.8)
    ax.set_xlabel("principal component"); ax.set_ylabel("% of plasma-parameter variance")
    ax.set_title(f"Plasma-parameter PCA scree\n{n90} PCs cover 90%, {n95} cover 95%")
    ax.set_xticks(x); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  -> {out}")


def _loadings(Vt, evr, params, out):
    k = min(5, Vt.shape[0])
    L = Vt[:k]
    fig, ax = plt.subplots(figsize=(1.1 * len(params) + 2, 0.7 * k + 2.5))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_yticks(range(k)); ax.set_yticklabels([f"PC{i+1}\n{evr[i]*100:.0f}%" for i in range(k)])
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([short_param(c) for c in params], rotation=45, ha="right", fontsize=8)
    for i in range(k):
        for j in range(len(params)):
            ax.text(j, i, f"{L[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(L[i, j]) > 0.45 else "black")
    ax.set_title("PCA loadings: how each component weights the plasma parameters")
    fig.colorbar(im, ax=ax, shrink=0.8, label="loading")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  -> {out}")


def _pcr(cum_by_p, idx, out):
    fig, ax = plt.subplots(figsize=(8, 5.3))
    for p, cumr2 in cum_by_p.items():
        x = np.arange(1, len(cumr2) + 1)
        ax.plot(x, cumr2, "-o", color=PCOLOR[p], label=SHORT[p])
    ax.set_xlabel("number of principal components (linear combos of plasma params)")
    ax.set_ylabel("cumulative R^2 predicting coil cost")
    ax.set_title(f"How many combinations of plasma parameters predict coil cost (i={idx})")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()