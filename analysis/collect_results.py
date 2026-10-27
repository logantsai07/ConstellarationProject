"""collect_results.py
Goes through the study output folder and puts every saved .npy solve into one
row of a csv.

Run in the same conda env as the study (out_dict can have jax arrays in it so
jax has to be importable):

    python collect_results.py \
        --input-dir /scratch/lct9592/output_constellaration_nfp=3 \
        --out /scratch/lct9592/all_results.csv

Older result files without the fingerprint/check fields just get blank columns.
"""

import argparse
import csv
import os
from os.path import join, isdir, isfile

import numpy as np


# Metrics quadcoil writes into out_dict (value_only=True). Any extra metric
# found in a file is added as a column automatically.
KNOWN_METRICS = [
    "f_B",
    "f_K",
    "f_max_Phi",
    "f_l1_Phi",
    "f_max_K_dot_grad_K_cyl",
    "f_max_K2",
    "f_max_force_cyl",
    "f_huber_K",
]

# Coil family per problem, for convenient filtering in the sheet.
COIL_TYPE = {
    "nescoil_A": "baseline",
    "nescoil_B": "baseline_poloidal",
    "C_sparse_dipole": "dipole",
    "D_thin_dipole": "dipole",
    "F_low_max_force_dipole": "dipole",
    "I_low_curvature_filament": "filament",
    "K_low_max_force_filament": "filament",
}


def to_float(x):
    """Coerce a quadcoil output (float, jax/np array, or {'value':...}) to float."""
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("value", "val", "objective", "metric", "f", "data"):
            if k in x:
                return to_float(x[k])
        return None
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return None


def parse_name(fname):
    """
    From a result filename return (problem, target_index).
    'nescoil_A.npy'            -> ('nescoil_A', None)
    'C_sparse_dipole_i3.npy'   -> ('C_sparse_dipole', 3)
    """
    stem = fname[:-4] if fname.endswith(".npy") else fname
    if stem.startswith("nescoil"):
        return stem, None
    if "_i" in stem:
        head, _, tail = stem.rpartition("_i")
        if tail.isdigit():
            return head, int(tail)
    return stem, None


def row_from_file(config_id, fname, path):
    d = np.load(path, allow_pickle=True).item()

    problem, target_index = parse_name(fname)
    out_dict    = d.get("out_dict", {}) or {}
    check       = d.get("check", {}) or {}
    fp          = d.get("fingerprint", {}) or {}
    status      = d.get("status", {}) or {}

    row = {
        "config_id":       config_id,
        "problem":         problem,
        "coil_type":       COIL_TYPE.get(problem, ""),
        "target_index":    target_index if target_index is not None else "",
        "objective_name":  fp.get("objective_name", ""),
        "objective_unit":  fp.get("objective_unit", ""),
        "fB_target":       fp.get("fB_target", check.get("fB_target", "")),
        # convergence / feasibility (from check when present, else top-level/status)
        "converged":       check.get("converged", d.get("converged", "")),
        "feasible":        check.get("feasible", ""),
        "severity":        check.get("severity", ""),
        "niter":           check.get("niter", status.get("niter", status.get("inner_fin_niter", ""))),
        "maxiter":         check.get("maxiter", ""),
        "fin_f":           check.get("fin_f", status.get("fin_f", status.get("inner_fin_f", ""))),
        "time_s":          d.get("time", ""),
        # config-level context from the fingerprint
        "nfp":                 fp.get("nfp", ""),
        "plasma_coil_distance": fp.get("plasma_coil_distance", ""),
        "mpol":                fp.get("mpol", ""),
        "ntor":                fp.get("ntor", ""),
        "net_poloidal_current_amperes": fp.get("net_poloidal_current_amperes", ""),
        # baseline-only extras
        "K_theta_avg":       d.get("K_theta_avg", ""),
        "K_theta_cons":      d.get("K_theta_cons", ""),
        "K_theta_sign_frac": d.get("K_theta_sign_frac", ""),
        # provenance
        "quadcoil_version":  fp.get("quadcoil_version", ""),
        "study_version":     fp.get("study_version", ""),
        "timestamp_utc":     d.get("timestamp_utc", ""),
        "file":              fname,
    }

    # every metric actually present in out_dict
    metrics_here = list(KNOWN_METRICS)
    for k in out_dict.keys():
        if k not in metrics_here:
            metrics_here.append(k)
    for m in metrics_here:
        row[m] = to_float(out_dict.get(m))

    return row


def main():
    p = argparse.ArgumentParser(description="Flatten quadcoil study .npy results into one CSV")
    p.add_argument("--input-dir", required=True,
                   help="output_constellaration_nfp=X directory")
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    rows = []
    n_files = 0
    n_bad = 0
    configs = sorted(d for d in os.listdir(args.input_dir)
                     if isdir(join(args.input_dir, d)))

    for config_id in configs:
        cdir = join(args.input_dir, config_id)
        for fname in sorted(os.listdir(cdir)):
            if not fname.endswith(".npy") or fname == "quadcoil_inputs.npy":
                continue
            path = join(cdir, fname)
            if not isfile(path):
                continue
            n_files += 1
            try:
                rows.append(row_from_file(config_id, fname, path))
            except Exception as e:
                n_bad += 1
                print(f"  could not read {path}: {e}")

    if not rows:
        print(f"No result .npy files found under {args.input_dir}")
        return

    # stable, readable column order; any stray keys appended at the end
    lead = [
        "config_id", "problem", "coil_type", "target_index",
        "objective_name", "objective_unit", "fB_target",
        "f_B", "f_K", "f_max_Phi", "f_l1_Phi",
        "f_max_K_dot_grad_K_cyl", "f_max_K2", "f_max_force_cyl", "f_huber_K",
        "converged", "feasible", "severity", "niter", "maxiter", "fin_f",
        "time_s", "nfp", "plasma_coil_distance", "mpol", "ntor",
        "net_poloidal_current_amperes",
        "K_theta_avg", "K_theta_cons", "K_theta_sign_frac",
        "quadcoil_version", "study_version", "timestamp_utc", "file",
    ]
    allkeys = set().union(*(r.keys() for r in rows))
    fieldnames = lead + sorted(k for k in allkeys if k not in lead)

    def sort_key(r):
        return (r["config_id"], str(r["problem"]),
                r["target_index"] if isinstance(r["target_index"], int) else -1)
    rows.sort(key=sort_key)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows from {n_files} files "
          f"({len(configs)} configs, {n_bad} unreadable) -> {args.out}")


if __name__ == "__main__":
    main()