"""
quadcoil_study.py
-----------------
Constellaration study script.

Implemented problems: A, B (nescoil baselines) and C, D, F, I, K.
(E, G, H, J are not implemented.)

Cached results are only reused when their saved fingerprint matches the current
run's parameters. Bump STUDY_VERSION whenever the problem set or the target
construction changes in a way that invalidates results on disk.

Usage
-----
    python quadcoil_study.py --input-dir ../output_constellaration_nfp=3
    python quadcoil_study.py --input-dir ../output_constellaration_nfp=3 --plasma-config-id <id>
    python quadcoil_study.py --input-dir ../output_constellaration_nfp=3 --coil-distance-fraction 0.6
"""

import argparse
import hashlib
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from os.path import join

import numpy as np
import jax
import jax.numpy as jnp

import quadcoil
from quadcoil.quantity import K_theta

try:
    from quadcoil.surface import SurfaceRZFourierJAX
except ImportError as _exc:
    raise ImportError(
        "Could not import SurfaceRZFourierJAX from quadcoil.surface. Check the "
        "installed layout with:\n"
        "  python -c \"import quadcoil, pkgutil; "
        "print([m.name for m in pkgutil.iter_modules(quadcoil.__path__)])\""
    ) from _exc

jax.config.update("jax_compilation_cache_dir", os.environ.get("QUADCOIL_JAX_CACHE", os.path.expanduser("~/.cache/quadcoil-jax")))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# Bump when the problem set, the fB target construction, or the meaning of a
# saved file changes. Old results then fail validation and are recomputed.
# 3: plasma_coil_distance is derived per config from the minor radius rather
#    than fixed at 1.4, which self-intersected the winding surface.
STUDY_VERSION = 3

# Documented quadcoil defaults, used only to interpret `niter` when --maxiter is
# not passed. The docstring and the code disagree on slsqp (200 vs 500), so this
# is advisory: an unrecognised solver marks `converged=None`, never a hard failure.
# For 'auglag-lbfgs' this cap is the OUTER loop; maxiter_inner defaults to 500.
DEFAULT_SOLVER = "auglag-lbfgs"
SOLVER_MAXITER_DEFAULT = {"auglag-lbfgs": 10000, "ipm": 100, "slsqp": 500}

# niter < maxiter is only a heuristic: a solver can stop early without
# converging. If the installed quadcoil exposes an explicit flag under any of
# these names, it wins over the heuristic. The observed status keys are printed
# once per run so this can be confirmed against the installed commit.
#
# Names that could plausibly hold a magnitude rather than a flag are excluded
# on purpose: a field called 'error' is far more likely to be a KKT residual or
# constraint-violation estimate, which is nonzero on every successful solve and
# would mark the entire run as failed.
SUCCESS_FLAG_KEYS = ("success", "converged", "solved", "is_converged")
FAILURE_FLAG_KEYS = ("failed", "diverged", "is_error")

# Below this fraction of K_theta samples agreeing with the sign of the average,
# the >= / <= choice in problem B is only weakly determined by the data.
K_THETA_SIGN_FRAC_WARN = 0.9

# Objectives minimised by the C/D/F/I/K problems. Every one must also appear in
# METRICS, because the nescoil baselines supply their unit scaling.
OBJECTIVES = (
    "f_l1_Phi",
    "f_max_Phi",
    "f_max_force_cyl",
    "f_max_K_dot_grad_K_cyl",
)

METRICS = (
    "f_B",
    "f_K",
    "f_max_Phi",
    "f_l1_Phi",
    "f_max_K_dot_grad_K_cyl",
    "f_max_K2",
    "f_max_force_cyl",
)

assert set(OBJECTIVES) <= set(METRICS), "every objective must be recorded as a metric"


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc).isoformat()


def to_float(x):
    """
    Coerce a quadcoil output to a python float, or None.

    quadcoil returns metrics as {'value': ...} (and {'value':, 'grad':} when
    value_only=False), hence the dict unwrapping.
    """
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


def _fmt(val, spec):
    return format(val, spec) if val is not None else "N/A"


def _status_get(status, *names, default=None):
    """
    Read a solver status field, tolerating the pre- and post-June-2026 quadcoil
    naming ('niter'/'fin_f' vs 'inner_fin_niter'/'inner_fin_f').
    """
    for n in names:
        if status and n in status:
            return status[n]
    return default


_STATUS_KEYS_LOGGED = False


def log_status_keys_once(status):
    """
    Print the solver status keys the first time we see them. Resolves which
    quadcoil status convention the installed commit uses without guessing.
    """
    global _STATUS_KEYS_LOGGED
    if _STATUS_KEYS_LOGGED or not status:
        return
    _STATUS_KEYS_LOGGED = True
    print(f"  solver status keys: {sorted(status.keys())}", flush=True)


def _as_flag(v):
    """
    Interpret a status value as a boolean flag, or None if it does not look
    like one. Name-based matching alone is not enough: a float under a
    flag-shaped name is far more likely to be a magnitude, so only genuine
    booleans and the exact values 0/1 are accepted.
    """
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    f = to_float(v)
    if f is not None and f in (0.0, 1.0):
        return bool(f)
    return None


def explicit_success(status):
    """
    Read an explicit success/failure flag from the solver status, or None if
    the status carries no such field. Falls through to the niter heuristic.
    """
    for k in SUCCESS_FLAG_KEYS:
        if status and k in status:
            b = _as_flag(status[k])
            if b is not None:
                return b
            _note_rejected_flag(k, status[k])
    for k in FAILURE_FLAG_KEYS:
        if status and k in status:
            b = _as_flag(status[k])
            if b is not None:
                return not b
            _note_rejected_flag(k, status[k])
    return None


_REJECTED_FLAGS_NOTED = set()


def _note_rejected_flag(key, value):
    """Report once per key: a flag-shaped name holding a non-boolean value."""
    if key in _REJECTED_FLAGS_NOTED:
        return
    _REJECTED_FLAGS_NOTED.add(key)
    print(f"  NOTE: solver status has '{key}' = {value!r}, which is not a boolean "
          f"flag; ignoring it and falling back to the niter heuristic.", flush=True)


def _clean_status(status):
    """
    Reduce a solver status dict to serialisable scalars.

    status['fin_x'] is the full flattened solution vector; stringifying it would
    put a truncated, useless array repr in every result file. Arrays are replaced
    by a shape note instead.
    """
    cleaned = {}
    for k, v in (status or {}).items():
        f = to_float(v)
        if f is not None:
            cleaned[k] = f
            continue
        shape = getattr(v, "shape", None)
        if shape is not None and np.prod(shape or (1,)) != 1:
            cleaned[k] = f"<array shape={tuple(shape)} dtype={getattr(v, 'dtype', '?')}>"
        else:
            s = str(v)
            cleaned[k] = s if len(s) <= 200 else s[:200] + "..."
    return cleaned


def _array_sha(a):
    if a is None:
        return None
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _quadcoil_version():
    try:
        import importlib.metadata as md
        return md.version("quadcoil")
    except Exception:
        return str(getattr(quadcoil, "__version__", "unknown"))


# ----------------------------------------------------------------------------
# fingerprinting: what makes a cached result valid
# ----------------------------------------------------------------------------

FINGERPRINT_KEYS = (
    "nfp", "stellsym", "plasma_stellsym", "mpol", "ntor",
    "plasma_mpol", "plasma_ntor",
    "net_poloidal_current_amperes", "net_toroidal_current_amperes",
    "plasma_coil_distance",
)


def make_fingerprint(kwargs_base, n_targets, solver, maxiter, extra=None):
    """
    Every input that changes the meaning of a saved result.

    n_targets matters because build_fB_targets spaces its multipliers over
    [1, 10] with n points: the same `_i1.npy` filename denotes a different
    target under a different n_targets.
    """
    fp = {
        "study_version":    STUDY_VERSION,
        "quadcoil_version": _quadcoil_version(),
        "n_targets":        int(n_targets),
        "metrics":          tuple(METRICS),
        "solver":           solver,
        "maxiter":          maxiter,
    }
    for k in FINGERPRINT_KEYS:
        v = kwargs_base.get(k)
        if isinstance(v, bool) or v is None:
            fp[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            fp[k] = float(v)
        else:
            fp[k] = str(v)
    fp["plasma_dofs_sha"]     = _array_sha(kwargs_base.get("plasma_dofs"))
    fp["Bnormal_plasma_sha"]  = _array_sha(kwargs_base.get("Bnormal_plasma"))
    if extra:
        fp.update(extra)
    return fp


def fingerprint_mismatch(saved, expected):
    """Return the name of the first differing key, or None if they agree."""
    if not isinstance(saved, dict):
        return "<missing fingerprint>"
    for k, want in expected.items():
        got = saved.get(k, "<absent>")
        if isinstance(want, float) and isinstance(got, float):
            if not math.isclose(want, got, rel_tol=1e-9, abs_tol=0.0):
                return k
        elif isinstance(want, tuple):
            if tuple(got) != want:
                return k
        elif got != want:
            return k
    return None


# ----------------------------------------------------------------------------
# solution validation
# ----------------------------------------------------------------------------

def check_solution(out_dict, status, solver, maxiter, fB_target=None,
                   fB_rtol=1e-6, is_boundary=False):
    """
    Decide whether a returned solution should be trusted, and how serious a
    failure is.

    A quadcoil call that exhausts its iteration limit still returns a solution;
    without this the result is cached and later treated as valid.

    severity:
      'ok'       -- usable
      'boundary' -- infeasible or non-converged at the i==0 target, which
                    constrains f_B to the nescoil optimum itself and is
                    expected to be marginal. Reported, does not fail the job.
      'hard'     -- anything else: nonfinite anywhere, or infeasible /
                    non-converged at i>0. Fails the job.
      'unknown'  -- convergence could not be determined from the status.
    """
    niter = to_float(_status_get(status, "niter", "inner_fin_niter"))
    fin_f = to_float(_status_get(status, "fin_f", "inner_fin_f"))
    # solver=None means the quadcoil default, which is 'auglag-lbfgs'.
    cap   = maxiter if maxiter is not None else SOLVER_MAXITER_DEFAULT.get(solver or DEFAULT_SOLVER)

    flag = explicit_success(status)
    if flag is not None:
        converged, converged_source = flag, "flag"
    elif niter is None or cap is None:
        converged, converged_source = None, None
    else:
        converged, converged_source = bool(niter < cap), "niter"

    finite = True
    if fin_f is not None and not math.isfinite(fin_f):
        finite = False
    for name in METRICS:
        v = to_float(out_dict.get(name))
        if v is not None and not math.isfinite(v):
            finite = False

    fB = to_float(out_dict.get("f_B"))
    if fB is None or fB_target is None:
        feasible = None
    else:
        feasible = bool(fB <= fB_target * (1.0 + fB_rtol))

    if not finite:
        severity = "hard"                       # never exempt, even at i==0
    elif converged is False or feasible is False:
        severity = "boundary" if is_boundary else "hard"
    elif converged is None:
        severity = "unknown"
    else:
        severity = "ok"

    return {
        "converged":        converged,
        "converged_source": converged_source,
        "feasible":         feasible,
        "finite":           finite,
        "niter":            niter,
        "maxiter":          cap,
        "fin_f":            fin_f,
        "f_B":              fB,
        "fB_target":        fB_target,
        "severity":         severity,
        "ok":               severity in ("ok", "unknown"),
    }


def describe_check(check):
    bits = []
    if check["converged"] is False:
        src = check["converged_source"]
        if src == "flag":
            bits.append("solver reported failure")
        else:
            bits.append(f"hit maxiter ({_fmt(check['niter'], '.0f')}/{check['maxiter']})")
    if check["feasible"] is False:
        bits.append(f"f_B={_fmt(check['f_B'], '.4g')} > target {_fmt(check['fB_target'], '.4g')}")
    if not check["finite"]:
        bits.append("nonfinite values")
    if check["converged"] is None:
        bits.append("convergence undetermined (no recognised status field)")
    return "; ".join(bits) if bits else "ok"


# ----------------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------------

def save_result(path, out_dict, dofs, elapsed, status, fingerprint, check, **extra):
    """
    Save one solve. qp is deliberately excluded -- it is not serialisable, so
    nothing downstream may depend on it.
    """
    payload = {
        "out_dict":      out_dict,
        "dofs":          np.array(dofs),
        "time":          elapsed,
        "timestamp_utc": now(),
        "status":        _clean_status(status),
        "fingerprint":   fingerprint,
        "check":         check,
    }
    payload.update(extra)
    np.save(path, payload, allow_pickle=True)
    print(f"    -> saved {path}")


def load_result(path):
    return np.load(path, allow_pickle=True).item()


def load_valid_cached(path, expected_fp, redo_unchecked=False, label=""):
    """
    Return a cached result only if it is trustworthy: fingerprint matches, the
    required objectives are present, and (optionally) its own check passed.
    Returns None when the caller should recompute.
    """
    if not os.path.exists(path):
        return None

    try:
        d = load_result(path)
    except Exception as e:
        print(f"    {label}unreadable cache ({e}) -- recomputing", flush=True)
        return None

    bad = fingerprint_mismatch(d.get("fingerprint"), expected_fp)
    if bad is not None:
        print(f"    {label}cache invalid: '{bad}' changed -- recomputing", flush=True)
        return None

    out_dict = d.get("out_dict", {})
    missing = [n for n in OBJECTIVES if to_float(out_dict.get(n)) is None]
    if missing:
        print(f"    {label}cache missing objectives {missing} -- recomputing", flush=True)
        return None

    if redo_unchecked and not d.get("check", {}).get("ok", True):
        print(f"    {label}cached result failed validation -- recomputing", flush=True)
        return None

    return d


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def _expected_surface_ndofs(mpol, ntor, stellsym):
    """
    simsopt SurfaceRZFourier.get_dofs() length. Stellarator-symmetric only:
    rc has (mpol+1)(2*ntor+1) - ntor entries, zs one fewer.
    Returns None for the non-symmetric layout rather than guessing.
    """
    if not stellsym:
        return None
    return 2 * (mpol + 1) * (2 * ntor + 1) - 2 * ntor - 1


def estimate_minor_radius(nfp, stellsym, plasma_dofs, plasma_mpol=4, plasma_ntor=4,
                          n_quad=128):
    """
    Estimate the plasma minor radius from its boundary DOFs.

    Averages the half-extents in R and Z over the sampled surface. This is a
    heuristic: R.max() - R.min() also picks up excursion of the magnetic axis,
    so for a strongly shaped stellarator it tends to overestimate `a`. It only
    has to be good enough to keep the offset winding surface from folding
    through the axis, which a fixed distance did not.

    Sampling [0, 1) in phi covers one field period; by periodicity the extrema
    are the same as over the full torus.
    """
    qp_phi   = jnp.linspace(0.0, 1.0, n_quad, endpoint=False)
    qp_theta = jnp.linspace(0.0, 1.0, n_quad, endpoint=False)
    s = SurfaceRZFourierJAX(
        nfp=int(nfp),
        stellsym=bool(stellsym),
        mpol=int(plasma_mpol),
        ntor=int(plasma_ntor),
        quadpoints_phi=qp_phi,
        quadpoints_theta=qp_theta,
        dofs=jnp.array(plasma_dofs),
    )
    g = np.asarray(s.gamma())
    R = np.sqrt(g[..., 0] ** 2 + g[..., 1] ** 2)
    Z = g[..., 2]
    a_R = (R.max() - R.min()) / 2.0
    a_Z = (Z.max() - Z.min()) / 2.0
    return float((a_R + a_Z) / 2.0)


def load_quadcoil_inputs(npy_path, mpol, ntor, coil_distance_fraction,
                         plasma_mpol, plasma_ntor):
    data = np.load(npy_path, allow_pickle=True).item()

    stellsym = bool(data["stellsym"])
    # quadcoil separates coil symmetry (stellsym) from plasma symmetry
    # (plasma_stellsym, default True). Prefer a distinct value if the inputs
    # ever carry one; otherwise the plasma inherits the coil setting, which is
    # still better than silently defaulting an asymmetric plasma to True.
    plasma_stellsym = bool(data.get("plasma_stellsym", stellsym))
    plasma_dofs = np.array(data["plasma_dofs"])
    nfp = int(data["nfp"])

    # Prefer values stored with the inputs if a future writer adds them.
    plasma_mpol = int(data.get("plasma_mpol", plasma_mpol))
    plasma_ntor = int(data.get("plasma_ntor", plasma_ntor))

    expected = _expected_surface_ndofs(plasma_mpol, plasma_ntor, plasma_stellsym)
    if expected is not None and len(plasma_dofs) != expected:
        print(
            f"  WARNING: plasma_dofs has {len(plasma_dofs)} entries but "
            f"plasma_mpol={plasma_mpol}, plasma_ntor={plasma_ntor}, "
            f"plasma_stellsym={plasma_stellsym} implies {expected}. "
            f"Pass --plasma-mpol/--plasma-ntor if this config was generated "
            f"at a different resolution."
        )

    # A fixed plasma_coil_distance is not portable across configs: at ~3x the
    # minor radius the uniform offset self-intersects and folds through the
    # magnetic axis, and quadcoil fails in root-finding before optimising.
    # Scale it to each plasma instead. The plasma surface uses plasma_stellsym.
    minor_r = estimate_minor_radius(
        nfp, plasma_stellsym, plasma_dofs, plasma_mpol, plasma_ntor
    )
    if not math.isfinite(minor_r) or minor_r <= 0:
        raise RuntimeError(
            f"estimated minor radius is {minor_r}; cannot derive a coil distance"
        )
    plasma_coil_distance = coil_distance_fraction * minor_r
    print(f"  minor_radius = {minor_r:.4g}  "
          f"coil_distance_fraction = {coil_distance_fraction:.3g}  "
          f"-> plasma_coil_distance = {plasma_coil_distance:.4g}")
    if coil_distance_fraction >= 1.0:
        print(f"  WARNING: coil_distance_fraction >= 1 offsets the winding surface "
              f"by at least a full minor radius; this is the regime where "
              f"uniform_offset() self-intersects.")

    kwargs_base = dict(
        nfp=nfp,
        stellsym=stellsym,
        plasma_stellsym=plasma_stellsym,
        mpol=mpol,
        ntor=ntor,
        plasma_dofs=plasma_dofs,
        plasma_mpol=plasma_mpol,
        plasma_ntor=plasma_ntor,
        net_poloidal_current_amperes=float(data["net_poloidal_current_amperes"]),
        net_toroidal_current_amperes=0.0,
        plasma_coil_distance=plasma_coil_distance,
    )

    if data.get("Bnormal_plasma") is not None:
        kwargs_base["Bnormal_plasma"] = np.array(data["Bnormal_plasma"])

    return kwargs_base


def solver_kwargs(solver, maxiter):
    """Only pass solver options that were explicitly requested."""
    kw = {}
    if solver is not None:
        kw["solver"] = solver
    if maxiter is not None:
        kw["maxiter"] = maxiter
    return kw


# ----------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------

def _objective_unit(out_dict, obj_name):
    u = to_float(out_dict.get(obj_name))
    if u is None:
        raise RuntimeError(f"'{obj_name}' missing from the nescoil baseline metrics")
    if not math.isfinite(u):
        raise RuntimeError(f"nescoil baseline gave a nonfinite '{obj_name}'")
    # A zero unit would silently destroy the problem scaling.
    return max(abs(u), 1e-12)


def run_nescoil(config_dir, kwargs_base, base_fp, solver, maxiter, redo_unchecked):
    """
    Problem A: pure Nescoil (minimise f_B).
    Problem B: Nescoil with a K_theta sign constraint.

    Returns the two out_dicts rather than qp/dofs: qp is not serialisable, so
    anything the sweep needs must come from the saved out_dict, which makes the
    cached and fresh paths identical.
    """
    path_A = join(config_dir, "nescoil_A.npy")
    path_B = join(config_dir, "nescoil_B.npy")

    fp_A = dict(base_fp, problem="nescoil_A")
    d = load_valid_cached(path_A, fp_A, redo_unchecked, label="[A] ")

    if d is not None:
        out_A = d["out_dict"]
        K_theta_avg = float(d["K_theta_avg"])
        K_theta_cons = str(d["K_theta_cons"])
        print("  [A] cached", flush=True)
        print(f"      f_B = {_fmt(to_float(out_A.get('f_B')), '.6g')}")
        print(f"      K_theta_avg = {K_theta_avg:.4g}  -> K_theta {K_theta_cons} 0")
    else:
        print("  [A] Nescoil (minimise f_B) ...", flush=True)
        t0 = time.perf_counter()
        out_A, qp_A, dofs_A, status_A = quadcoil.quadcoil(
            objective_name="f_B",
            objective_unit=1.0,
            metric_name=METRICS,
            value_only=True,
            **solver_kwargs(solver, maxiter),
            **kwargs_base,
        )
        elapsed_A = time.perf_counter() - t0
        print(f"      f_B = {_fmt(to_float(out_A.get('f_B')), '.6g')}   ({elapsed_A:.1f} s)")

        check_A = check_solution(out_A, status_A, solver, maxiter)
        log_status_keys_once(status_A)
        _fBA = to_float(out_A.get("f_B"))
        if _fBA is None or not np.isfinite(_fBA) or _fBA <= 0:
            raise RuntimeError(f"nescoil A produced no usable f_B (value={_fBA}): {describe_check(check_A)}")
        if not check_A["ok"]:
            print(f"      WARNING: baseline A not converged ({describe_check(check_A)}), using f_B={_fBA:.6g} as unit anyway")

        # K_theta is not in METRICS, so check_A does not cover it. A nonfinite
        # average silently poisons everything downstream: max(nan, 1.0) is nan
        # in python, so K_theta_unit would become nan and take the B and
        # filament constraint scalings with it.
        K_theta_arr = np.asarray(K_theta(qp_A, dofs_A))
        K_theta_avg = float(np.average(K_theta_arr))
        if not math.isfinite(K_theta_avg):
            raise RuntimeError(f"nescoil A gave nonfinite K_theta_avg={K_theta_avg}")
        if not np.all(np.isfinite(K_theta_arr)):
            raise RuntimeError("nescoil A gave nonfinite K_theta samples")

        K_theta_cons = ">=" if K_theta_avg >= 0 else "<="
        # How well-determined that sign choice is. Near 0.5 the average is a
        # cancellation artefact and the >= / <= choice is close to arbitrary,
        # which flips the entire filament branch for this config.
        sign_frac = float(np.mean(np.sign(K_theta_arr) == np.sign(K_theta_avg)))
        print(f"      K_theta_avg = {K_theta_avg:.4g}  -> K_theta {K_theta_cons} 0"
              f"  (sign agreement {sign_frac:.2f})")
        if sign_frac < K_THETA_SIGN_FRAC_WARN:
            print(f"      WARNING: only {sign_frac:.0%} of K_theta samples share the sign of "
                  f"the average; the '{K_theta_cons}' choice for problem B is weakly "
                  f"determined for this config.")

        save_result(path_A, out_A, dofs_A, elapsed_A, status_A, fp_A, check_A,
                    K_theta_avg=K_theta_avg, K_theta_cons=K_theta_cons,
                    K_theta_sign_frac=sign_frac)

    fB_A = to_float(out_A.get("f_B"))
    if fB_A is None or not math.isfinite(fB_A) or fB_A <= 0:
        raise RuntimeError(f"nescoil A gave an unusable f_B={fB_A}; cannot build targets")

    K_theta_unit = max(abs(K_theta_avg), 1.0)

    # K_theta_cons and K_theta_unit are both part of B's problem statement, so
    # both belong in B's fingerprint: if A is recomputed and either the sign
    # flips or the scaling shifts, B is stale.
    fp_B = dict(base_fp, problem="nescoil_B",
                K_theta_cons=K_theta_cons,
                K_theta_unit=float(K_theta_unit))
    d = load_valid_cached(path_B, fp_B, redo_unchecked, label="[B] ")

    if d is not None:
        out_B = d["out_dict"]
        print("  [B] cached", flush=True)
        print(f"      f_B = {_fmt(to_float(out_B.get('f_B')), '.6g')}")
    else:
        print("  [B] Nescoil w/ K_theta sign constraint ...", flush=True)
        t0 = time.perf_counter()
        out_B, qp_B, dofs_B, status_B = quadcoil.quadcoil(
            objective_name="f_B",
            objective_unit=1.0,
            constraint_name=("K_theta",),
            constraint_type=(K_theta_cons,),
            constraint_value=np.array([0.0]),
            constraint_unit=(K_theta_unit,),
            metric_name=METRICS,
            value_only=True,
            **solver_kwargs(solver, maxiter),
            **kwargs_base,
        )
        elapsed_B = time.perf_counter() - t0
        print(f"      f_B = {_fmt(to_float(out_B.get('f_B')), '.6g')}   ({elapsed_B:.1f} s)")

        check_B = check_solution(out_B, status_B, solver, maxiter)
        _fBB = to_float(out_B.get("f_B"))
        if _fBB is None or not np.isfinite(_fBB) or _fBB <= 0:
            raise RuntimeError(f"nescoil B produced no usable f_B (value={_fBB}): {describe_check(check_B)}")
        if not check_B["ok"]:
            print(f"      WARNING: baseline B not converged ({describe_check(check_B)}), using f_B={_fBB:.6g} as unit anyway")

        save_result(path_B, out_B, dofs_B, elapsed_B, status_B, fp_B, check_B,
                    K_theta_avg=K_theta_avg, K_theta_cons=K_theta_cons)

    fB_B = to_float(out_B.get("f_B"))
    if fB_B is None or not math.isfinite(fB_B) or fB_B <= 0:
        raise RuntimeError(f"nescoil B gave an unusable f_B={fB_B}; cannot build targets")

    return fB_A, fB_B, out_A, out_B, K_theta_cons, K_theta_avg


def build_fB_targets(fB_A, fB_B, n=5):
    # The first multiplier is 1.0, i.e. i=0 constrains f_B <= the unconstrained
    # nescoil optimum itself. That point is feasible only on the boundary and
    # will often exhaust maxiter for reasons unrelated to the solver -- report
    # it separately when auditing convergence.
    scale = 10 ** np.linspace(0, 1, n)
    return scale * fB_A, scale * fB_B


def make_kwargs_list(K_theta_cons, K_theta_avg):
    K_theta_unit = max(abs(K_theta_avg), 1.0)

    filament_extra = (
        ("K_theta",),
        (K_theta_cons,),
        np.array([0.0]),
        (K_theta_unit,),
    )

    def dipole(label, obj):
        return dict(label=label, target_key="dipole",
                    objective_name=obj, extra_constraints=None)

    def filament(label, obj):
        return dict(label=label, target_key="filament",
                    objective_name=obj, extra_constraints=filament_extra)

    return [
        dipole("C_sparse_dipole",        "f_l1_Phi"),
        dipole("D_thin_dipole",          "f_max_Phi"),
        dipole("F_low_max_force_dipole", "f_max_force_cyl"),
        filament("I_low_curvature_filament", "f_max_K_dot_grad_K_cyl"),
        filament("K_low_max_force_filament", "f_max_force_cyl"),
    ]


# ----------------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------------

def run_one_config(config_dir, kwargs_base, n_targets, solver, maxiter, redo_unchecked):
    """
    Returns (n_errors, counts) where counts tallies solves by severity:
    'hard' (job-failing), 'boundary' (expected marginality at i==0), and
    'unknown' (convergence undeterminable from the solver status).
    """
    base_fp = make_fingerprint(kwargs_base, n_targets, solver, maxiter)

    fB_A, fB_B, out_A, out_B, K_theta_cons, K_theta_avg = run_nescoil(
        config_dir, kwargs_base, base_fp, solver, maxiter, redo_unchecked
    )

    print(f"  fB_A={fB_A:.4g}  fB_B={fB_B:.4g}")

    fB_target_dipole, fB_target_filament = build_fB_targets(fB_A, fB_B, n=n_targets)
    target_arrays = {"dipole": fB_target_dipole, "filament": fB_target_filament}
    baseline_out  = {"dipole": out_A,            "filament": out_B}

    n_errors = 0
    counts = {"boundary": 0, "hard": 0, "unknown": 0}
    K_theta_unit = max(abs(K_theta_avg), 1.0)

    for kwarg_i in make_kwargs_list(K_theta_cons, K_theta_avg):
        label      = kwarg_i["label"]
        target_key = kwarg_i["target_key"]
        obj_name   = kwarg_i["objective_name"]
        fB_targets = target_arrays[target_key]
        extra      = kwarg_i["extra_constraints"]
        obj_unit   = _objective_unit(baseline_out[target_key], obj_name)

        print(f"\n  [{label}]  objective={obj_name}  unit={obj_unit:.4g}", flush=True)

        for i in range(n_targets):
            fB_target_i = float(fB_targets[i])
            save_path   = join(config_dir, f"{label}_i{i}.npy")
            is_boundary = (i == 0)

            # obj_unit and the target are part of the problem statement, so a
            # changed baseline invalidates the cached result even if every
            # global parameter is unchanged. The filament problems also carry
            # the K_theta constraint's direction and scaling.
            fp_i = dict(
                base_fp,
                problem=label,
                target_index=i,
                fB_target=fB_target_i,
                objective_name=obj_name,
                objective_unit=float(obj_unit),
                K_theta_cons=K_theta_cons if extra is not None else None,
                K_theta_unit=float(K_theta_unit) if extra is not None else None,
            )

            cached = load_valid_cached(save_path, fp_i, redo_unchecked, label=f"i={i} ")
            if cached is not None:
                sev = cached.get("check", {}).get("severity", "ok")
                flag = "" if sev == "ok" else f"  [{sev.upper()}]"
                print(f"    i={i}: cached, skipping.{flag}")
                if sev in counts:
                    counts[sev] += 1
                continue

            print(f"    i={i}  fB_target={fB_target_i:.4g}", end="  ", flush=True)

            if extra is None:
                c_names  = ("f_B",)
                c_types  = ("<=",)
                c_values = np.array([fB_target_i])
                c_units  = (fB_target_i,)
            else:
                ex_names, ex_types, ex_values, ex_units = extra
                c_names  = ("f_B",)  + ex_names
                c_types  = ("<=",)   + ex_types
                c_values = np.concatenate([[fB_target_i], ex_values])
                c_units  = (fB_target_i,) + ex_units

            try:
                t0 = time.perf_counter()
                out_dict, qp, dofs, status = quadcoil.quadcoil(
                    objective_name=obj_name,
                    objective_unit=obj_unit,
                    constraint_name=c_names,
                    constraint_type=c_types,
                    constraint_value=c_values,
                    constraint_unit=c_units,
                    metric_name=METRICS,
                    value_only=True,
                    **solver_kwargs(solver, maxiter),
                    **kwargs_base,
                )
                elapsed = time.perf_counter() - t0

                note = " (JIT compile + run)" if i == 0 else ""
                print(f"({elapsed:.1f} s{note})")

                check = check_solution(out_dict, status, solver, maxiter,
                                       fB_target=fB_target_i, is_boundary=is_boundary)
                log_status_keys_once(status)
                print(f"      f_B={_fmt(check['f_B'], '.4g')}  "
                      f"fin_f={_fmt(check['fin_f'], '.4f')}  "
                      f"niter={_fmt(check['niter'], '.0f')}/{check['maxiter']}")
                if check["severity"] != "ok":
                    print(f"      {check['severity'].upper()}: {describe_check(check)}")
                    counts[check["severity"]] += 1

                save_result(save_path, out_dict, dofs, elapsed, status, fp_i, check)

            except Exception as e:
                print(f"\n      FAILED: {e}")
                traceback.print_exc()
                n_errors += 1

    return n_errors, counts


def find_config_dirs(input_dir):
    configs = []
    for name in sorted(os.listdir(input_dir)):
        subdir = join(input_dir, name)
        if os.path.isdir(subdir) and os.path.exists(join(subdir, "quadcoil_inputs.npy")):
            configs.append((name, subdir))
    return configs


def run_study(input_dir, mpol, ntor, n_targets, coil_distance_fraction,
              plasma_mpol, plasma_ntor, solver, maxiter, redo_unchecked,
              plasma_config_id=None, task_id=0, num_tasks=1):
    configs = find_config_dirs(input_dir)

    if not configs:
        raise RuntimeError(f"No quadcoil_inputs.npy files found under {input_dir}")

    if plasma_config_id is not None:
        configs = [(n, d) for n, d in configs if n == plasma_config_id]
        if not configs:
            raise RuntimeError(f"plasma_config_id '{plasma_config_id}' not found in {input_dir}")
    else:
        configs = [c for i, c in enumerate(configs) if i % num_tasks == task_id]

    print(f"quadcoil {_quadcoil_version()}, study version {STUDY_VERSION}")
    print(f"Found {len(configs)} config(s) to process.")

    n_config_failed = 0
    n_errors_total = 0
    totals = {"boundary": 0, "hard": 0, "unknown": 0}

    for idx, (config_id, config_dir) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {idx+1}/{len(configs)}: {config_id}")
        print(f"{'='*60}")

        npy_path = join(config_dir, "quadcoil_inputs.npy")
        try:
            kwargs_base = load_quadcoil_inputs(
                npy_path, mpol, ntor, coil_distance_fraction, plasma_mpol, plasma_ntor
            )
        except Exception as e:
            print(f"  Failed to load inputs: {e} -- skipping.")
            traceback.print_exc()
            n_config_failed += 1
            continue

        try:
            n_err, counts = run_one_config(
                config_dir, kwargs_base, n_targets, solver, maxiter, redo_unchecked
            )
            n_errors_total += n_err
            for k in totals:
                totals[k] += counts[k]
            if n_err or counts["hard"]:
                n_config_failed += 1
        except Exception as e:
            print(f"  Config failed: {e} -- continuing to next.")
            traceback.print_exc()
            n_config_failed += 1

    n_ok = len(configs) - n_config_failed
    print(f"\n\nDone. {n_ok}/{len(configs)} configs clean, "
          f"{n_config_failed} with failures, "
          f"{n_errors_total} solve errors, "
          f"{totals['hard']} hard validation failures, "
          f"{totals['boundary']} boundary (i=0) suspects, "
          f"{totals['unknown']} undetermined.")

    if totals["unknown"]:
        print("WARNING: convergence could not be determined for "
              f"{totals['unknown']} solve(s). The installed quadcoil exposes no "
              "status field this script recognises -- see the 'solver status "
              "keys' line above and extend _status_get / SUCCESS_FLAG_KEYS.")

    # Non-zero exit so a SLURM array task that produced errors is not recorded
    # as a success. Boundary suspects at i==0 are expected and do not fail the
    # job; nonfinite results and any failure at i>0 do.
    return 1 if (n_config_failed or n_errors_total or totals["hard"]) else 0


def main():
    p = argparse.ArgumentParser(
        description="quadcoil constellaration study - problems A, B, C, D, F, I, K"
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--plasma-config-id", default=None)
    p.add_argument("--mpol",        type=int,   default=8)
    p.add_argument("--ntor",        type=int,   default=8)
    p.add_argument("--plasma-mpol", type=int,   default=4)
    p.add_argument("--plasma-ntor", type=int,   default=4)
    p.add_argument("--n-targets",   type=int,   default=5)
    p.add_argument("--coil-distance-fraction", type=float, default=0.6,
                   help="coil distance as a fraction of each plasma's estimated "
                        "minor radius (replaces the old fixed --plasma-coil-distance)")
    p.add_argument("--solver",  default=None,
                   help="quadcoil solver name; omit to use the library default")
    p.add_argument("--maxiter", type=int, default=None,
                   help="iteration cap; omit to use the library default")
    p.add_argument("--redo-unchecked", action="store_true",
                   help="recompute cached results that failed validation")
    p.add_argument("--task-id",   type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        p.error(f"--input-dir '{args.input_dir}' is not a directory")
    if args.n_targets < 1:
        p.error("--n-targets must be >= 1")
    if args.num_tasks < 1:
        p.error("--num-tasks must be >= 1")
    if not (0 <= args.task_id < args.num_tasks):
        p.error(f"--task-id must satisfy 0 <= task_id < num_tasks "
                f"(got {args.task_id}, num_tasks={args.num_tasks})")
    if args.mpol < 1 or args.ntor < 1:
        p.error("--mpol and --ntor must be >= 1")
    if args.plasma_mpol < 1 or args.plasma_ntor < 1:
        p.error("--plasma-mpol and --plasma-ntor must be >= 1")
    if not math.isfinite(args.coil_distance_fraction) or args.coil_distance_fraction <= 0:
        p.error("--coil-distance-fraction must be finite and positive")
    if args.maxiter is not None and args.maxiter < 1:
        p.error("--maxiter must be >= 1")

    return run_study(
        input_dir=args.input_dir,
        mpol=args.mpol,
        ntor=args.ntor,
        n_targets=args.n_targets,
        coil_distance_fraction=args.coil_distance_fraction,
        plasma_mpol=args.plasma_mpol,
        plasma_ntor=args.plasma_ntor,
        solver=args.solver,
        maxiter=args.maxiter,
        redo_unchecked=args.redo_unchecked,
        plasma_config_id=args.plasma_config_id,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    sys.exit(main())