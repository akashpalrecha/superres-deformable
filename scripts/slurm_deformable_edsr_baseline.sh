#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 3-0:00
#SBATCH -p seas_gpu
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
python main.py --model deformable_edsr --res_scale 0.1 --scale 2 --ext sep --save_results --save deformable_edsr_baseline_x2
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""
echo "---- BEGIN TESTING ----"
echo ""
python main.py --model deformable_edsr --res_scale 0.1 --scale 2 --ext sep --save_results --save deformable_edsr_baseline_x2/results --pre_train ../experiment/deformable_edsr_baseline_x2/model/model_best.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
echo ""
echo "---- TESTING COMPLETE ----"
echo ""