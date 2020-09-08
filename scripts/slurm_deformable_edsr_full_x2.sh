#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 2-6:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/deformable_edsr_baseline_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/deformable_edsr_baseline_x2_%j.err

echo "Settinp up env. Loading Conda, etc ..."
source ~/.bashrc
cd /n/home00/apalrecha/lab
echo "Activating EDSR conda environment ..."
source activate envs/edsr
echo "Loading cuda/10.1.243-fasrc01 ..."
module load cuda/10.1.243-fasrc01
echo "Setup Done!"
cd Deblurring/EDSR-PyTorch/src/
echo ""
echo "---- BEGIN TRAINING ----"
echo ""
python main.py --model deformable_edsr --scale 2 --save deformable_edsr_full_x2 --ext sep --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900 --save_results
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""
