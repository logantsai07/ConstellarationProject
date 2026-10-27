#!/bin/bash -l
#SBATCH --job-name=quadcoil_full3
#SBATCH --account=torch_pr_292_courant
#SBATCH --partition=h200_courant
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=/scratch/lct9592/logs/full3_%A_%a.out
#SBATCH --error=/scratch/lct9592/logs/full3_%A_%a.err
#
# NOTE: /scratch/lct9592/logs must exist BEFORE the first sbatch. Slurm opens
# these files before the script body runs, so an in-script mkdir is too late --
# the task dies at launch with nothing written anywhere. Run once:
#     mkdir -p /scratch/lct9592/logs
#
# --requeue is safe now: results are fingerprinted, so a preempted task that
# restarts skips completed solves instead of redoing or corrupting them.

set -uo pipefail

SCRATCH_DIR=/scratch/lct9592
INPUT_DIR="$SCRATCH_DIR/output_constellaration_nfp=3"
PY="$SCRATCH_DIR/conda_envs/constellaration/bin/python"
SCRIPT="$SCRATCH_DIR/quadcoil_study2.py"

# quadcoil_study2.py hardcodes its JAX cache as "../jax-caches", relative to the
# working directory. With no cd, that resolved against wherever you happened to
# run sbatch from -- e.g. /scratch/jax-caches, which is not writable, so JAX
# warns and silently runs with no persistent cache. Running from $SCRATCH_DIR/run
# puts it at /scratch/lct9592/jax-caches for every task, every submission.
RUN_DIR="$SCRATCH_DIR/run"a
mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || { echo "cannot cd to $RUN_DIR" >&2; exit 2; }

module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate constellaration

# The striping is `i % num_tasks == task_id`, so the array must start at 0.
# A 1-N array orphans config 0 and trips the argparse bound on the last task.
if [[ "${SLURM_ARRAY_TASK_MIN:-0}" -ne 0 ]]; then
    echo "ERROR: --array must start at 0 (got min=${SLURM_ARRAY_TASK_MIN})." >&2
    exit 2
fi

# Fail loudly instead of falling back to CPU. On 100 configs a silent CPU
# fallback is the difference between one night and one week, discovered at the
# 12-hour timeout.
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Start time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --num-tasks "$SLURM_ARRAY_TASK_COUNT"
status=$?

echo "-----------------------------------------------------------"
echo "End time: $(date)"
echo "exit=$status"

# Must be the last command. A trailing `echo "Job completed!"` here would make
# every task exit 0 and report COMPLETED to sacct regardless of what the study
# script found, which defeats the whole point of the exit-code handling.
exit $status

# ---------------------------------------------------------------------------
# Sizing: --array=0-4 is 5 tasks over 100 configs = 20 configs each, 540 solves
# per task in 12 h. That needs a mean solve under ~80 s. Time the smoke test
# first; if a config runs long, widen to --array=0-9 or 0-19 -- num-tasks tracks
# SLURM_ARRAY_TASK_COUNT automatically, so the array line is the only edit.
#
# Smoke test one config first (also warms ../jax-caches with the four distinct
# compilations: unconstrained A, one-constraint B and dipole, two-constraint
# filament -- so the array does not cold-compile the same shapes on 5 H200s):
#
#   srun --account=torch_pr_292_courant --partition=h200_courant \
#        --gres=gpu:1 --time=01:00:00 --cpus-per-task=4 --mem=32000 --pty bash
#   cd /scratch/lct9592/run
#   /scratch/lct9592/conda_envs/constellaration/bin/python \
#       /scratch/lct9592/quadcoil_study2.py \
#       --input-dir /scratch/lct9592/output_constellaration_nfp=3 \
#       --plasma-config-id <id>
#   echo "exit=$?"
#
# After the array:
#   sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed | grep -v COMPLETED
#   grep -h "^Done" /scratch/lct9592/logs/full3_<jobid>_*.out
# ---------------------------------------------------------------------------
