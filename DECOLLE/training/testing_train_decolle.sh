#!/bin/bash
path="../decolle"
output_path="../output_data"
#--resume_from=
#params:
#params_dvsgestures_torchneuromorphic.yml
#params_dvsgestures_01.yml
#--save_dir=
#--verbose=True to display the parameters we are using
parameters_to_check=(01 02 03 04 05 06 07)
for t in ${parameters_to_check[@]}; do
    mkdir ./plots/bar/$t
    mkdir ./plots/confusion_matrices/$t
    mkdir ./plots/correct_perc/$t
    mkdir ./plots/test_loss/$t
    mkdir ./plots/total_loss/$t
done
for t in ${parameters_to_check[@]}; do
    python $path/scripts/train_lenet_decolle.py --params_file=$path/scripts/parameters/params_dvsgestures_${t}.yml --save_dir=$output_path/$t
done
python plot_results.py
