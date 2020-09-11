#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-20:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/one_deformable_edsr_baseline_x2_100_compare_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/one_deformable_edsr_baseline_x2_100_compare_%j.err

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
echo "---- NOW TRAINING DEFORMABLE EDSR ----"
python main.py --model deformable_edsr --res_scale 0.1 --scale 2 --save one_deformable_edsr_baseline_x2_100 --ext sep --save_results --one_deformable_block
echo ""
echo ""
echo "---- NOW TRAINING PLAIN EDSR ----"
python main.py --model EDSR            --res_scale 0.1 --scale 2 --save edsr_baseline_x2_100                --ext sep --save_results
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""
echo "---- BEGIN TESTING ----"
echo ""
python main.py --model deformable_edsr --res_scale 0.1 --scale 2 --one_deformable_block --save one_deformable_edsr_baseline_x2_100/results --pre_train ../experiment/one_deformable_edsr_baseline_x2_100/model/model_best.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900 --ext sep --save_results
python main.py --model EDSR --res_scale 0.1 --scale 2 --save edsr_baseline_x2_100/results --pre_train ../experiment/edsr_baseline_x2_100/model/model_best.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900 --ext sep --save_results
echo ""
echo "---- TESTING COMPLETE ----"
echo ""