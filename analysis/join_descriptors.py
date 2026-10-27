"""join_descriptors.py
Adds the ConStellaration descriptors (iota, aspect ratio, elongation, mirror
ratio, vacuum well, ...) to all_results.csv by plasma_config_id.

quadcoil_inputs.npy doesn't have these, so they get pulled again from the
HuggingFace "default" split. Run on torch in the env with `datasets`:

    python join_descriptors.py \
        --results /scratch/lct9592/all_results.csv \
        --out     /scratch/lct9592/all_results_with_descriptors.csv

Streams the dataset, keeps every scalar field for the config ids we need
(flattened to metrics.*, boundary.*, ...), stops once they're all found, and
left-merges onto the results. Prints the column list at the end.
"""

import argparse
import sys

import numpy as np
import pandas as pd


def flatten(obj, prefix=""):
    """Recursively flatten nested dicts into dotted keys; keep leaves as-is."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    else:
        out[prefix] = obj
    return out


def is_scalar(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    # numpy 0-d / python None are handled elsewhere
    return False


def scalar_fields(example):
    """From one streamed example, return {dotted_key: scalar_value} plus the id."""
    flat = flatten(example)
    out = {}
    for k, v in flat.items():
        if v is None:
            continue
        if is_scalar(v):
            out[k] = float(v) if not isinstance(v, bool) else bool(v)
    return flat, out


def find_id(flat):
    """Locate the plasma config id regardless of nesting."""
    for key in ("plasma_config_id", "metadata.plasma_config_id", "misc.plasma_config_id"):
        if key in flat and flat[key] is not None:
            return str(flat[key])
    # last resort: any key ending in plasma_config_id
    for k, v in flat.items():
        if k.endswith("plasma_config_id") and v is not None:
            return str(v)
    return None


def main():
    ap = argparse.ArgumentParser(description="Join ConStellaration descriptors onto all_results.csv")
    ap.add_argument("--results", required=True, help="all_results.csv")
    ap.add_argument("--out", required=True, help="output CSV with descriptor columns")
    ap.add_argument("--dataset", default="proxima-fusion/constellaration")
    ap.add_argument("--config", default="default", help='HF config name (default: "default")')
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="stop after scanning this many rows (0 = no limit / until all found)")
    a = ap.parse_args()

    res = pd.read_csv(a.results)
    if "config_id" not in res.columns:
        sys.exit("all_results.csv has no 'config_id' column")
    needed = set(res["config_id"].astype(str))
    print(f"all_results.csv: {len(res)} rows, {len(needed)} unique config_id")

    try:
        from datasets import load_dataset
    except Exception as e:
        sys.exit(f"could not import datasets ({e}); run this in the download_constellaration env")

    # open the stream (try with config name, fall back to without)
    try:
        stream = load_dataset(a.dataset, a.config, split=a.split, streaming=True)
    except Exception as e:
        print(f"  load_dataset(config={a.config!r}) failed ({e}); retrying without config name")
        stream = load_dataset(a.dataset, split=a.split, streaming=True)

    found = {}            # config_id -> {dotted scalar fields}
    all_scalar_keys = set()
    id_missing = 0
    scanned = 0

    for ex in stream:
        scanned += 1
        flat, scal = scalar_fields(ex)
        pid = find_id(flat)
        if pid is None:
            id_missing += 1
        elif pid in needed and pid not in found:
            found[pid] = scal
            all_scalar_keys.update(scal.keys())
        if scanned % 5000 == 0:
            print(f"  scanned {scanned:>7d} rows, matched {len(found)}/{len(needed)}")
        if len(found) == len(needed):
            print(f"  all {len(needed)} configs found after scanning {scanned} rows")
            break
        if a.max_scan and scanned >= a.max_scan:
            print(f"  hit --max-scan={a.max_scan}; stopping")
            break

    print(f"\nscanned {scanned} rows total; matched {len(found)}/{len(needed)} config_ids")
    if id_missing:
        print(f"  ({id_missing} streamed rows had no recognizable plasma_config_id)")

    if not found:
        print("\nNO config_id matched plasma_config_id. Likely the folder names are not the "
              "plasma_config_id. Samples to compare:")
        print("  all_results config_id:", list(needed)[:5])
        # show what ids the stream actually has
        sample_ids = []
        for ex in load_dataset(a.dataset, a.config, split=a.split, streaming=True):
            fid = find_id(flatten(ex))
            if fid:
                sample_ids.append(fid)
            if len(sample_ids) >= 5:
                break
        print("  dataset plasma_config_id:", sample_ids)
        sys.exit("Fix the key mapping and rerun.")

    desc = pd.DataFrame.from_dict(found, orient="index")
    desc.index.name = "config_id"
    desc = desc.reset_index()
    desc["config_id"] = desc["config_id"].astype(str)

    # keep a tidy, useful subset first, then everything else
    descriptor_cols = sorted(c for c in desc.columns if c != "config_id")
    print(f"\n{len(descriptor_cols)} scalar descriptor columns pulled:")
    for c in descriptor_cols:
        print("   ", c)

    res["config_id"] = res["config_id"].astype(str)
    merged = res.merge(desc, on="config_id", how="left", suffixes=("", "_desc"))

    cov = merged[descriptor_cols[0]].notna().mean() if descriptor_cols else 0.0
    print(f"\nmerged: {len(merged)} rows; descriptor coverage ~{cov:.0%} of rows")
    merged.to_csv(a.out, index=False)
    print(f"-> {a.out}")

    # also drop a compact per-config descriptor table for quick eyeballing
    side = a.out.rsplit(".", 1)[0] + "_descriptors_only.csv"
    desc.to_csv(side, index=False)
    print(f"-> {side}  (one row per config_id, descriptors only)")


if __name__ == "__main__":
    main()