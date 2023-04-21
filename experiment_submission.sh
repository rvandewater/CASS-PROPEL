#!/bin/bash
export PARTITION=hpcpu
export GPUS=0
export CPUS=8

export prefix=111111111

sbatch slurm_submission.sh ${prefix}1
sbatch slurm_submission.sh ${prefix}2
sbatch slurm_submission.sh ${prefix}3
sbatch slurm_submission.sh ${prefix}4
sbatch slurm_submission.sh ${prefix}5
sbatch slurm_submission.sh ${prefix}6
sbatch slurm_submission.sh ${prefix}7
sbatch slurm_submission.sh ${prefix}8
sbatch slurm_submission.sh ${prefix}9
sbatch slurm_submission.sh ${prefix}11
