#!/bin/bash
#SBATCH --job-name=sklearn_analysis_%j
#SBATCH --mail-type=Fail,End
#SBATCH --partition=compute # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=50gb
#SBATCH --gpus=0
#SBATCH --output=logs/%x_%a_%j_cass_preop_elect.log # %x is job-name, %j is job id, %a is array id
#SBATCH --time 48:00:00

eval "$(/opt/conda/bin/conda shell.bash hook)" # Conda initialization in the bash shell
conda activate propel


echo "This is a SLURM job named" $SLURM_JOB_NAME "with array id" $SLURM_ARRAY_TASK_ID "and job id" $SLURM_JOB_ID
echo "Resources allocated: " $SLURM_CPUS_PER_TASK "CPUs, " $SLURM_MEM_PER_NODE "GB RAM, " $SLURM_GPUS_PER_NODE "GPU"
echo $1


python3 main.py cass_preop_emerg --seed $1 --cores=$SLURM_CPUS_PER_TASK --out_dir=cass_preop_emergency

#python3 complete_evaluation.py cass_preop_elect --seed $1 --out_dir=cass_preop_elect

#python3 complete_evaluation.py cass_lab --seed $1 --out_dir=cass_lab



