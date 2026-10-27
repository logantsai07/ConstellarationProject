import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

A = "/mnt/user-data/attachments/22f51c76-c7d8-564b-9905-15120c27db37"
OUT = "/tmp/claude-0/-home-claude/22f51c76-c7d8-564b-9905-15120c27db37/scratchpad"
FIGS = "/tmp/claude-0/-home-claude/22f51c76-c7d8-564b-9905-15120c27db37/scratchpad/poster/figs"
SRC = {"easy": f"{A}/cfc9f6eb-easy_phi.npz", "hard": f"{A}/7aee8e53-hard_phi.npz"}
LABEL = {"easy": "Low mirror ratio  (easy to coil)", "hard": "High mirror ratio  (hard to coil)"}
D = {k: np.load(v) for k, v in SRC.items()}


def smooth_phi(phi, fac=10):
    trend = phi.mean(axis=1, keepdims=True)
    resid = zoom(phi - trend, fac, order=3, mode="grid-wrap")
    trend_f = zoom(trend.ravel(), fac, order=3, mode="nearest")[:, None]
    return resid + trend_f


def draw(ax, k, vmax, levels_line):
    d = D[k]
    phi = smooth_phi(np.asarray(d["phi"]))
    th = np.linspace(0, 2 * np.pi, phi.shape[1])
    ze = np.linspace(0, 2 * np.pi / int(d["nfp"]), phi.shape[0])
    Z = phi.T / 1e6
    cf = ax.contourf(ze, th, Z, levels=40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(ze, th, Z, levels=levels_line, colors="k", linewidths=1.2)
    ax.set_title(LABEL[k], fontsize=13)
    ax.set_xticks([0, np.pi/3, 2*np.pi/3]); ax.set_xticklabels(["0", r"$\pi/3$", r"$2\pi/3$"])
    ax.set_yticks([0, np.pi, 2*np.pi]); ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    r = np.ptp(d["phi"] - d["phi"].mean(axis=1, keepdims=True)) / abs(float(d["G"]))
    ax.text(0.03, 0.97, f"ripple / net current = {r:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    return cf


G = np.mean([abs(float(D[k]["G"])) for k in D])
vmax = max(np.abs(D[k]["phi"]).max() for k in D) / 1e6
levels_line = np.arange(-3 * G, 3 * G, G / 10.0) / 1e6

# ---- horizontal (for slides / viewing) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
for ax, k in zip(axes, ["easy", "hard"]):
    cf = draw(ax, k, vmax, levels_line)
    ax.set_xlabel(r"toroidal angle  $\zeta$", fontsize=11)
axes[0].set_ylabel(r"poloidal angle  $\theta$", fontsize=11)
cb = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.02); cb.set_label(r"$\Phi$  (MA)", fontsize=11)
fig.suptitle("Current potential on the winding surface; black lines are coils", fontsize=14, y=1.06)
fig.savefig(f"{OUT}/phi_compare.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", f"{OUT}/phi_compare.png")

# ---- vertical (for the poster column) ----
figv, axv = plt.subplots(2, 1, figsize=(6.4, 7.0), constrained_layout=True)
for ax, k in zip(axv, ["easy", "hard"]):
    cf = draw(ax, k, vmax, levels_line)
    ax.set_ylabel(r"poloidal $\theta$", fontsize=11)
axv[1].set_xlabel(r"toroidal angle  $\zeta$  (one field period)", fontsize=11)
cb = figv.colorbar(cf, ax=axv, shrink=0.8, pad=0.02, location="right"); cb.set_label(r"$\Phi$  (MA)", fontsize=11)
figv.savefig(f"{FIGS}/phi_compare_v.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", f"{FIGS}/phi_compare_v.png")
