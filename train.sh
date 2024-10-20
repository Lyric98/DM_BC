#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=DMBC
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=experiments/FFHQ512.out
#SBATCH --account=pr_174_general

python sr.py -p train -c config/sr_wave_64_512FFHQ.json
