import sys, os, argparse
sys.path.insert(0, "/scratch/lct9592")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist", type=float, default=1.4,
                    help="coil_distance_fraction (render used 0.6; scan default ~1.4)")
    ap.add_argument("--ncoils", type=int, default=3, help="coils_per_half_period")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cfg = os.path.join(a.input_dir, a.config_id)

    import quadcoil
    import quadcoil_study as qs
    kw = qs.load_quadcoil_inputs(os.path.join(cfg, "quadcoil_inputs.npy"), 8, 8, a.dist, 4, 4)
    print(f"  solving f_B at coil_distance_fraction={a.dist}, ncoils={a.ncoils} ...")
    _out, qp, dofs, _st = quadcoil.quadcoil(objective_name='f_B', metric_name=('f_B'),
                                            value_only=True, **kw)

    # coilability signal: net poloidal current magnitude
    for attr in ("net_poloidal_current_amperes", "net_poloidal_current", "G"):
        if hasattr(qp, attr):
            print(f"  {attr} = {float(getattr(qp, attr)):.4e}")
            break

    from quadcoil.io.coil_cutting import simsopt_coil_from_qp
    coils = None
    for od in (10, 8, 6):                      # never fall back to the garbage order-2 fit
        try:
            coils = simsopt_coil_from_qp(qp, dofs, a.ncoils, order=od)
            print(f"  cut at Fourier order {od}")
            break
        except AssertionError:
            print(f"  order {od} too high for a short contour; lowering")
    if coils is None:
        raise SystemExit("  NOT cleanly coilable at order>=6 at this distance "
                         "-- raise --dist or lower --ncoils, or this config is genuinely hard to coil")

    curves = [getattr(c, "curve", c) for c in coils]
    pts = [np.asarray(c.gamma()) for c in curves]

    # flag coils that actually encircle the plasma (true modular coils)
    allp = np.vstack(pts)
    R0 = np.hypot(allp[:, 0], allp[:, 1]).mean()
    Z0 = allp[:, 2].mean()
    def wraps(c):
        r = np.hypot(c[:, 0], c[:, 1])
        ang = np.arctan2(c[:, 2] - Z0, r - R0)
        return abs(np.diff(np.unwrap(np.r_[ang, ang[0]])).sum()) / (2 * np.pi) > 0.6
    mask = np.array([wraps(p) for p in pts])
    print(f"  cut {len(pts)} contours; {int(mask.sum())} wrap the plasma (modular)")

    np.savez(a.out + "_coils.npz", scal=np.arange(len(pts), dtype=float), n=len(pts),
             modular=mask, **{f"c{i}": p for i, p in enumerate(pts)})
    print("  wrote", a.out + "_coils.npz")


if __name__ == "__main__":
    main()
