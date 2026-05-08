#!/bin/bash -l
#SBATCH --job-name=quadcoil_full
#SBATCH --account=torch_pr_292_courant
#SBATCH --partition=h200_courant
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --array=0-15
#SBATCH --output=/scratch/lct9592/logs/full_%A_%a.out
#SBATCH --error=/scratch/lct9592/logs/full_%A_%a.err

mkdir -p /scratch/lct9592/logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate constellaration

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Start time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

/scratch/lct9592/conda_envs/constellaration/bin/python /scratch/lct9592/quadcoil_study.py \
    --input-dir /scratch/lct9592/output_constellaration_nfp=3 \
    --n-targets 5 \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-tasks $SLURM_ARRAY_TASK_COUNT

echo "End time: $(date)"
echo "Job completed!"