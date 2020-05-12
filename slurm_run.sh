#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p cox
#SBATCH --gres=gpu:1
#SBATCH --mem=16384
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/edsr_baseline_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/edsr_baseline_x2_%j.err

module load Anaconda3/5.0.1-fasrc02
cd /n/home00/apalrecha/lab
source activate envs/edsr
cd Deblurring/EDSR-PyTorch/src/
python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2_%j --reset --ext sep
echo "Training complete!"
