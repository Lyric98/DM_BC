#!/bin/sh
#
#SBATCH --account=biostats        # Replace ACCOUNT with your group account name
#SBATCH --job-name=DMBC           # The job name.
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH --gres=gpu:2              # gpu number
#SBATCH -t 0-12:00                 # Runtime in D-HH:MM
#SBATCH -N 1                      # Nodes required for the job.
#SBATCH --mem-per-cpu=50gb        # The memory the job will use per cpu core
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yl5465@cumc.columbia.edu

 

python sr.py -p train -c config/sr_wave_64_512CBIS.json -enable_wandb -log_wandb_ckpt -log_eval