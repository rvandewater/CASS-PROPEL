#!/bin/bash
#SBATCH --job-name=sklearn_analysis
#SBATCH --partition=hpcpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --array=0-100%5
#SBATCH --output=slurm_logs/slurm-%A_%a.out

eval "$(conda shell.bash hook)"
conda activate sklearn

python3 complete_evaluation.py esophagus -v=True --seed ${SLURM_ARRAY_TASK_ID} --out_dir=final_results --correlation_threshold=0.85
