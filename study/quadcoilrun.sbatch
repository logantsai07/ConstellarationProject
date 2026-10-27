#!/bin/bash -l
#SBATCH --job-name=quadcoilrun
#SBATCH --account=torch_pr_292_courant
#SBATCH --partition=h200_courant
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/scratch/lct9592/logs/quadcoilrun_%A_%a.out
#SBATCH --error=/scratch/lct9592/logs/quadcoilrun_%A_%a.err


set -uo pipefail

SCRATCH_DIR=/scratch/lct9592
INPUT_DIR="$SCRATCH_DIR/output_constellaration_nfp=3"
PY="$SCRATCH_DIR/conda_envs/constellaration/bin/python"
SCRIPT="$SCRATCH_DIR/quadcoil_study.py"

# --------------------------------------------------------- sanity checks ----
[[ -x "$PY"        ]] || { echo "ERROR: no interpreter at $PY" >&2; exit 2; }
[[ -f "$SCRIPT"    ]] || { echo "ERROR: no script at $SCRIPT" >&2; exit 2; }
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: no input dir at $INPUT_DIR" >&2; exit 2; }

# make sure this isn't the old script with the fixed 1.4 distance (fails every config)
if ! grep -q "coil_distance_fraction" "$SCRIPT"; then
    echo "ERROR: $SCRIPT has no coil_distance_fraction, old copy?" >&2
    echo "       grep -l coil_distance_fraction $SCRATCH_DIR/quadcoil_study*.py" >&2
    exit 2
fi

# read task count with a default so set -u doesn't kill it
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-}"
if ! [[ "$NUM_TASKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_COUNT is '$NUM_TASKS'; submit with --array." >&2
    exit 2
fi

# configs are split by i % num_tasks == task_id so the array has to start at 0
if [[ "${SLURM_ARRAY_TASK_MIN:-0}" -ne 0 ]]; then
    echo "ERROR: --array must start at 0 (got min=${SLURM_ARRAY_TASK_MIN})." >&2
    exit 2
fi

# ---------------------------------------------------------------- paths ----
# the jax cache path is "../jax-caches" relative to cwd, so cd into run/ to put it
# at /scratch/lct9592/jax-caches (otherwise it can end up somewhere unwritable)
RUN_DIR="$SCRATCH_DIR/run"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || { echo "ERROR: cannot cd to $RUN_DIR" >&2; exit 2; }

module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate constellaration

# force GPU, don't let it fall back to CPU (way too slow for 100 configs)
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Array task: ${TASK_ID} / ${NUM_TASKS}"
echo "Start time: $(date)"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Script: $SCRIPT"
echo "Workdir: $(pwd)   (jax cache -> $(cd .. && pwd)/jax-caches)"
"$PY" - <<'PY'
import jax, quadcoil
print("jax:", jax.__version__, jax.devices())
print("quadcoil:", getattr(quadcoil, "__version__", "unknown"), quadcoil.__file__)
PY
echo "-----------------------------------------------------------"

"$PY" "$SCRIPT" \
    --input-dir "$INPUT_DIR" \
    --n-targets 5 \
    --coil-distance-fraction 0.6 \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS"
status=$?

echo "-----------------------------------------------------------"
echo "End time: $(date)"
echo "exit=$status"

exit $status

