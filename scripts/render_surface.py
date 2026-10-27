import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
from scipy.ndimage import zoom
import pyvista as pv

A = "/mnt/user-data/attachments/22f51c76-c7d8-564b-9905-15120c27db37"
OUT = "/tmp/claude-0/-home-claude/22f51c76-c7d8-564b-9905-15120c27db37/scratchpad"
FIGS = f"{OUT}/poster/figs"
SRC = {"easy": f"{A}/cfc9f6eb-easy_phi.npz", "hard": f"{A}/7aee8e53-hard_phi.npz"}


def build(d, fac=5):
    xyz = np.asarray(d["xyz"]); phi = np.asarray(d["phi"]); nfp = int(d["nfp"])
    tm = phi.mean(axis=1)
    trend = np.polyval(np.polyfit(np.arange(len(tm)), tm, 1), np.arange(len(tm)))[:, None]
    ripple = np.tile(phi - trend, (nfp, 1)) / 1e6                 # MA, (96,34)
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    up = lambda a: zoom(a, fac, order=3, mode="grid-wrap")        # smooth, periodic both ways
    X, Y, Z, R = up(X), up(Y), up(Z), up(ripple)
    close = lambda a: np.concatenate([np.concatenate([a, a[:1]], 0),
                                      np.concatenate([a, a[:1]], 0)[:, :1]], 1)
    return close(X), close(Y), close(Z), close(R)


D = {k: np.load(v) for k, v in SRC.items()}
B = {k: build(D[k]) for k in D}
vmax = max(np.abs(B[k][3]).max() for k in B)
rad = max(np.max(np.sqrt((B[k][0]-B[k][0].mean())**2 + (B[k][1]-B[k][1].mean())**2
                         + (B[k][2]-B[k][2].mean())**2)) for k in B)


def render(tag):
    Xc, Yc, Zc, Rc = B[tag]
    grid = pv.StructuredGrid(Xc, Yc, Zc); grid["phi"] = Rc.ravel(order="F")
    ctr = np.array([Xc.mean(), Yc.mean(), Zc.mean()])
    pl = pv.Plotter(off_screen=True, window_size=(1700, 1300)); pl.set_background("white")
    pl.add_mesh(grid, scalars="phi", cmap="RdBu_r", clim=(-vmax, vmax), show_scalar_bar=False,
                smooth_shading=True, specular=0.2, specular_power=15)
    pl.enable_anti_aliasing("ssaa")
    dvec = np.array([1.0, -0.6, 0.55]); dvec = dvec/np.linalg.norm(dvec)
    pl.camera.position = tuple(ctr + dvec*rad*3.0)
    pl.camera.focal_point = tuple(ctr); pl.camera.up = (0, 0, 1)
    out = f"{OUT}/surf_{tag}.png"; pl.screenshot(out); pl.close(); return out


paths = {k: render(k) for k in B}
print("rendered", paths, "vmax(MA)=", round(float(vmax), 3))

# ---- composite: stacked, shared colorbar ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
import numpy as np

def crop(p):
    im = np.asarray(Image.open(p).convert("RGB"))
    nz = (im < 250).any(2); ys, xs = np.where(nz)
    return im[ys.min():ys.max()+1, xs.min():xs.max()+1]

figv, axv = plt.subplots(2, 1, figsize=(5.6, 7.6), constrained_layout=True)
titles = {"easy": "Low mirror ratio  (easy to coil)", "hard": "High mirror ratio  (hard to coil)"}
for ax, k in zip(axv, ["easy", "hard"]):
    ax.imshow(crop(paths[k])); ax.set_axis_off(); ax.set_title(titles[k], fontsize=14)
sm = ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"); sm.set_array([])
cb = figv.colorbar(sm, ax=axv, shrink=0.75, pad=0.02)
cb.set_label(r"current-potential structure  $\Phi'$  (MA)", fontsize=12)
figv.savefig(f"{FIGS}/surface_compare_v.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", f"{FIGS}/surface_compare_v.png")
