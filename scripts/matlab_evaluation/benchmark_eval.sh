#!/bin/bash

# {1} : scale. Ex: 2, 3, 4
# {2} : model output folder. Ex: benchmark_resutls_edsr_x2
# modify datasets_folder and results_folder according to your system if needed

scale=${1}
datasets_folder="/n/pfister_lab2/Lab/akash/Deblurring/datasets/benchmark/"
results_folder="/n/pfister_lab2/Lab/akash/Deblurring/EDSR-PyTorch/experiment/${2}/"
suffix="_x${scale}_SR"
get_eval() { echo "Evaluate_PSNR_SSIM '${datasets_folder}${1}/HR' '${results_folder}results-${1}' ${scale} ${suffix} '${results_folder}matlab_eval/${1}.txt' ${1} .png"; }

echo "Scale: ${scale}"
echo "Dataset Folder: ${datasets_folder}"
echo "Results folder: ${results_folder}"
echo "Suffix: ${suffix}"
echo "Sample command: $(get_eval sample)"

eval_set5=$(get_eval Set5)
eval_set14=$(get_eval Set14)
# eval_manga109=$(get_eval Manga109)
eval_urban100=$(get_eval Urban100)
eval_b100=$(get_eval B100)
eval_div2k=$(get_eval DIV2K)
matlab -nojvm -r "${eval_set5}; ${eval_set14}; ${eval_manga109}; ${eval_urban100}; ${eval_b100}; ${eval_div2k}; exit;"

