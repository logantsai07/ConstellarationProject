import sys, os, argparse
sys.path.insert(0, "/scratch/lct9592")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist", type=float, default=0.6,
                    help="coil_distance_fraction (0.6 solves for both configs)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cfg = os.path.join(a.input_dir, a.config_id)

    import quadcoil
    import quadcoil_study as qs
    from quadcoil.quantity import Phi_with_net_current

    kw = qs.load_quadcoil_inputs(os.path.join(cfg, "quadcoil_inputs.npy"), 8, 8, a.dist, 4, 4)
    print(f"  solving f_B at coil_distance_fraction={a.dist} ...")
    _out, qp, dofs, _st = quadcoil.quadcoil(objective_name='f_B', metric_name=('f_B'),
                                            value_only=True, **kw)

    phi = np.asarray(Phi_with_net_current(qp, dofs))          # [zeta, theta]
    theta = np.asarray(qp.quadpoints_theta) * 2 * np.pi
    zeta = np.asarray(qp.quadpoints_phi) * 2 * np.pi
    nfp = int(qp.nfp)

    G = np.nan
    for attr in ("net_poloidal_current_amperes", "net_poloidal_current", "G"):
        if hasattr(qp, attr):
            G = float(getattr(qp, attr)); break

    out = {"phi": phi, "theta": theta, "zeta": zeta, "nfp": nfp, "G": G}

    # bonus: winding-surface geometry on the same grid, for an optional 3D surface render
    try:
        surf = qp.winding_surface.to_simsopt()
        out["xyz"] = np.asarray(surf.gamma())                # [zeta, theta, 3]
        print("  winding surface xyz shape", out["xyz"].shape)
    except Exception as e:
        print("  (winding-surface geometry skipped:", e, ")")

    print(f"  phi shape {phi.shape}  theta {theta.shape}  zeta {zeta.shape}  nfp {nfp}  G {G:.3e}")
    np.savez(a.out + "_phi.npz", **out)
    print("  wrote", a.out + "_phi.npz")


if __name__ == "__main__":
    main()
