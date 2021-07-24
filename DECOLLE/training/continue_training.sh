#!/bin/bash
path="../decolle"
output_path="../output_data"
#--resume_from=
#params:
#params_dvsgestures_torchneuromorphic.yml
#params_dvsgestures_01.yml
#--save_dir=
#--verbose=True to display the parameters we are using

python $path/scripts/train_lenet_decolle.py --resume_from="logs/output_data/01/Jul21_19-53-16_george-System-Product-Name/" --params_file=$path/scripts/parameters/params_dvsgestures_01.yml --save_dir=$output_path/01

