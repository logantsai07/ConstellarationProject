"""
analyze_results.py
------------------
Parses SLURM output logs from quadcoil_study and plots histograms
of f_B values and convergence metrics across all configs and problems.

Usage
-----
    python analyze_results.py --log-dir /scratch/lct9592/logs --out-dir ./plots
    python analyze_results.py --log-dir /scratch/lct9592/logs --out-dir ./plots --prefix full_
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROBLEMS = [
    "C_sparse_dipole",
    "D_thin_dipole",
    "F_low_max_force_dipole",
    "I_low_curvature_filament",
    "K_low_max_force_filament",
]

NESCOIL = ["nescoil_A", "nescoil_B"]


def parse_logs(log_dir, prefix="full_"):
    """
    Parse all SLURM output logs matching prefix*.out in log_dir.
    Returns a dict:
        data[problem][i] = list of f_B values across configs
        data['nescoil_A'] = list of fB_A values
        data['nescoil_B'] = list of fB_B values
        convergence[problem][i] = list of (fin_f, niter) tuples
    """
    pattern = os.path.join(log_dir, f"{prefix}*.out")
    log_files = glob.glob(pattern)

    if not log_files:
        pattern = os.path.join(log_dir, "study_*.out")
        log_files = glob.glob(pattern)

    if not log_files:
        raise RuntimeError(f"No log files found in {log_dir} with prefix '{prefix}'")

    print(f"Found {len(log_files)} log file(s)")

    data = defaultdict(lambda: defaultdict(list))
    convergence = defaultdict(lambda: defaultdict(list))

    re_config    = re.compile(r"Config \d+/\d+: (\S+)")
    re_fB_A      = re.compile(r"fB_A=([\d.e+\-]+)")
    re_fB_B      = re.compile(r"fB_B=([\d.e+\-]+)")
    re_problem   = re.compile(r"\[(\w+)\]\s+objective=")
    re_result    = re.compile(r"i=(\d+)\s+fB_target=[\d.e+\-]+\s+\([\d.]+ s")
    re_fB        = re.compile(r"f_B=([\d.e+\-]+)")
    re_finf      = re.compile(r"fin_f=([\d.e+\-]+)")
    re_niter     = re.compile(r"niter=(\d+)")
    re_nescoil_A = re.compile(r"\[A\].*\n.*f_B = ([\d.e+\-]+)")
    re_nescoil_B = re.compile(r"\[B\].*\n.*f_B = ([\d.e+\-]+)")

    for log_file in log_files:
        with open(log_file) as f:
            content = f.read()
            lines = content.splitlines()

        current_problem = None
        current_i = None
        in_result = False

        for idx, line in enumerate(lines):
            m = re_fB_A.search(line)
            if m:
                data["nescoil_A"][0].append(float(m.group(1)))

            m = re_fB_B.search(line)
            if m:
                data["nescoil_B"][0].append(float(m.group(1)))

            m = re_problem.search(line)
            if m:
                current_problem = m.group(1)
                current_i = None
                in_result = False
                continue

            m = re_result.search(line)
            if m and current_problem:
                current_i = int(m.group(1))
                in_result = True
                continue

            if in_result and current_problem is not None and current_i is not None:
                mf = re_fB.search(line)
                mfi = re_finf.search(line)
                mn = re_niter.search(line)
                if mf and mfi and mn:
                    data[current_problem][current_i].append(float(mf.group(1)))
                    convergence[current_problem][current_i].append(
                        (float(mfi.group(1)), int(mn.group(1)))
                    )
                    in_result = False

    return data, convergence


def plot_histograms(data, convergence, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    all_problems = list(data.keys())

    for problem in all_problems:
        targets = sorted(data[problem].keys())
        if not targets:
            continue

        n_targets = len(targets)
        fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))
        if n_targets == 1:
            axes = [axes]

        fig.suptitle(f"{problem}", fontsize=13, fontweight="bold")

        for ax, i in zip(axes, targets):
            vals = np.array(data[problem][i])
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                ax.set_title(f"i={i}\nno data")
                continue

            log_vals = np.log10(np.abs(vals) + 1e-300)
            ax.hist(log_vals, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
            ax.set_xlabel("log10(|f_B|)")
            ax.set_ylabel("count")
            ax.set_title(f"i={i}  n={len(vals)}\nmedian={np.median(vals):.3g}")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"hist_{problem}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  saved {out_path}")

    plot_convergence(convergence, out_dir)
    plot_nescoil_summary(data, out_dir)


def plot_convergence(convergence, out_dir):
    problems = [p for p in convergence if p not in ("nescoil_A", "nescoil_B")]
    if not problems:
        return

    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4))
    if len(problems) == 1:
        axes = [axes]

    fig.suptitle("Convergence: fraction hitting niter=1000", fontsize=12)

    for ax, problem in zip(axes, problems):
        targets = sorted(convergence[problem].keys())
        fracs = []
        for i in targets:
            niters = [n for _, n in convergence[problem][i]]
            frac = sum(1 for n in niters if n >= 1000) / len(niters) if niters else 0
            fracs.append(frac)

        ax.bar(targets, fracs, color="tomato", edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_xlabel("fB target index i")
        ax.set_ylabel("fraction not converged")
        ax.set_title(problem.replace("_", "\n"), fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "convergence_summary.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def plot_nescoil_summary(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Nescoil baseline f_B distribution", fontsize=12)

    for ax, key, label in zip(axes,
                               ["nescoil_A", "nescoil_B"],
                               ["Problem A (pure Nescoil)", "Problem B (K_theta constrained)"]):
        vals = np.array(data[key][0])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) == 0:
            ax.set_title(f"{label}\nno data")
            continue
        ax.hist(np.log10(vals), bins=20, color="mediumseagreen", edgecolor="white", alpha=0.85)
        ax.set_xlabel("log10(f_B)")
        ax.set_ylabel("count")
        ax.set_title(f"{label}\nn={len(vals)}  median={np.median(vals):.3g}")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "hist_nescoil.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def print_summary(data, convergence):
    print("\n=== Summary ===")
    for problem in sorted(data.keys()):
        targets = sorted(data[problem].keys())
        total = sum(len(data[problem][i]) for i in targets)
        print(f"\n{problem}:")
        for i in targets:
            vals = np.array(data[problem][i])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                print(f"  i={i}: no data")
                continue
            conv = convergence.get(problem, {}).get(i, [])
            n_stuck = sum(1 for _, n in conv if n >= 1000)
            print(f"  i={i}: n={len(vals)}  "
                  f"median={np.median(vals):.3g}  "
                  f"min={np.min(vals):.3g}  "
                  f"max={np.max(vals):.3g}  "
                  f"not_converged={n_stuck}/{len(conv)}")


def main():
    p = argparse.ArgumentParser(description="Analyze quadcoil study SLURM logs")
    p.add_argument("--log-dir", default="/scratch/lct9592/logs")
    p.add_argument("--out-dir", default="./plots")
    p.add_argument("--prefix",  default="full_",
                   help="Log file prefix (default: full_). Use 'study_' for test runs.")
    args = p.parse_args()

    print(f"Parsing logs from {args.log_dir} with prefix '{args.prefix}'...")
    data, convergence = parse_logs(args.log_dir, prefix=args.prefix)

    print_summary(data, convergence)
    print(f"\nPlotting histograms to {args.out_dir}...")
    plot_histograms(data, convergence, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()