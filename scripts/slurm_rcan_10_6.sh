#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-23:50
#SBATCH -p seas_dgx1
#SBATCH --exclude=seasdgx103
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/rcan_10_6_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/EDSR-PyTorch/experiment/slurm_output/rcan_10_6_%j.err

echo "Settinp up env. Loading Conda, MATLAB, etc ..."
source ~/.bashrc
cd /n/home00/apalrecha/lab
echo "Activating EDSR conda environment ..."
source activate envs/edsr
echo "Loading MATLAB ..."
loadmatlab
# echo "Loading cuda/10.1.243-fasrc01 ..."
# module load cuda/10.1.243-fasrc01
echo "Setup Done!"
cd Deblurring/EDSR-PyTorch/src/
echo ""
echo "---- BEGIN TRAINING ----"
echo ""
python main.py --model rcan     --n_resgroups 6 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save rcan_10_6_baseline_x2     --epochs 100
echo ""
echo "---- TRAINING COMPLETE ----"

echo ""
echo "---- BEGIN TESTING ----"

echo ""
echo "Start generating results"
python main.py --model rcan     --n_resgroups 6 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save rcan_10_6_baseline_x2/results     --pre_train ../experiment/rcan_10_6_baseline_x2/model/model_best.pt     --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
echo ""

echo "Start MATLAB Testing"
echo ""
cd ../scripts/matlab_evaluation
./benchmark_eval.sh 2 rcan_10_6_baseline_x2/results
echo ""

echo "---- TESTING COMPLETE ----"
echo ""