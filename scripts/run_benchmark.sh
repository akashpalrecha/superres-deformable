#!/bin/bash
cd ../src
python main.py --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900 --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --save_results --save benchmark_edsr_x2 ${1}

python main.py --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900 --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --save_results --save benchmark_edsr_x3 ${1}

python main.py --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --save_results --save benchmark_edsr_x4 ${1}
