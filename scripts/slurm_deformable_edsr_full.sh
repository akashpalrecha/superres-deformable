#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 2-6:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/deformable_edsr_baseline_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/deformable_edsr_baseline_x2_%j.err

echo "Settinp up env. Loading Conda, etc..."
source ~/.bashrc
cd /n/home00/apalrecha/lab
echo "Activating EDSR conda environment"
source activate envs/edsr
echo "Done"
cd Deblurring/EDSR-PyTorch/src/
echo ""
echo "---- BEGIN TRAINING ----"
echo ""
python main.py --model deformable_edsr --scale 2 --patch_size 96 --save deformable_edsr_baseline_x2 --reset --ext sep
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""
