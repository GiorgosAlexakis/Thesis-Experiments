#!/bin/bash
path="."
output_path="./output_data"

parameters_to_check=(01 02 03 04 05 06 07 08 09 10)
#parameters_to_check=(10)

for t in ${parameters_to_check[@]}; do
    mkdir -p ./plots/bar/$t
    mkdir -p ./plots/confusion_matrices/$t
    mkdir -p ./plots/correct_perc/$t
    mkdir -p ./plots/test_loss/$t
    mkdir -p ./plots/total_loss/$t
    mkdir -p ./output_data/$t
    
done
for t in ${parameters_to_check[@]}; do
    echo "Starting training with parameters $t"
    python $path/dvs_gestures_classifier.py $path/parameters_${t}.yml $output_path/$t
    echo "Finished training with parameters $t"
    echo "Plotting training results for parameters $t"
    python plot_results.py $t
    echo "Finished plotting training results for parameters $t"
done

