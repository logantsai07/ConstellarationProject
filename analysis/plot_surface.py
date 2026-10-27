"""
plot_surface.py
Builds the plasma surface from quadcoil_inputs.npy, builds the uniform_offset
surface, plots both as 3D point clouds, and checks the offset sign.
"""
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import quadcoil
from quadcoil.surface import SurfaceRZFourierJAX


def plot_cloud(gamma, title, path):
    g = np.array(gamma).reshape(-1, 3)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(g[:, 0], g[:, 1], g[:, 2], s=2)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--distance", type=float, default=1.4)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    data = jnp.load(args.input, allow_pickle=True).item()
    nfp = int(data["nfp"])
    print("nfp:", nfp, "stellsym:", bool(data["stellsym"]))
    print("plasma_dofs shape:", np.array(data["plasma_dofs"]).shape)

    quadpoints_phi = jnp.linspace(0, 1.0, 64, endpoint=False)
    quadpoints_theta = jnp.linspace(0, 1.0, 64, endpoint=False)

    surface = SurfaceRZFourierJAX(
        nfp=nfp,
        stellsym=bool(data["stellsym"]),
        mpol=4, ntor=4,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        dofs=jnp.array(data["plasma_dofs"]),
    )
    print("Surface built OK")

    g_plasma = np.array(surface.gamma())
    plot_cloud(g_plasma, "Plasma surface", f"{args.out_dir}/plasma_surface.png")

    # Offset
    try:
        offset = surface.uniform_offset(args.distance)
        print("Offset surface built OK")
        g_offset = np.array(offset.gamma())
        plot_cloud(g_offset, f"Offset surface (d={args.distance})",
                   f"{args.out_dir}/offset_surface.png")

        r_plasma = np.sqrt(g_plasma[..., 0]**2 + g_plasma[..., 1]**2)
        r_offset = np.sqrt(g_offset[..., 0]**2 + g_offset[..., 1]**2)
        print(f"\nMean R plasma: {r_plasma.mean():.4f}")
        print(f"Mean R offset: {r_offset.mean():.4f}")
        print(f"Min dist plasma->axis: {r_plasma.min():.4f}")
        print(f"Min dist offset->axis: {r_offset.min():.4f}")

        # Direct point-to-point distance between the two surfaces
        d = np.sqrt(((g_offset - g_plasma)**2).sum(axis=-1))
        print(f"\nMean surface separation: {d.mean():.4f}  (requested {args.distance})")
        print(f"Min  surface separation: {d.min():.4f}")
        print(f"Max  surface separation: {d.max():.4f}")

        if r_offset.mean() < r_plasma.mean():
            print("\n*** WARNING: offset surface is SMALLER than plasma, SIGN FLIP ***")
        else:
            print("\nOffset surface is larger than plasma (expected direction).")
    except Exception as e:
        print("offset failed:", e)


if __name__ == "__main__":
    main()