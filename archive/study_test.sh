#!/bin/bash -l
#SBATCH --job-name=quadcoil_study
#SBATCH --account=torch_pr_292_courant
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --output=/scratch/lct9592/logs/study_%j.out
#SBATCH --error=/scratch/lct9592/logs/study_%j.err

mkdir -p /scratch/lct9592/logs

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate constellaration

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python /scratch/lct9592/quadcoil_study.py \
    --input-dir /scratch/lct9592/output_constellaration_nfp=3 \
    --plasma-config-id D22RYC5Z58coDhr9Wc6PXqz \
    --n-targets 5

echo "End time: $(date)"
echo "Job completed!"