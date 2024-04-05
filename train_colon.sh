#!/bin/bash


#settings
root_folder=''
data_folder="./colon_30"
roi_x=128   
roi_y=128
roi_z=32
v_min=1
v_max=255

data_dir="$root_folder""/$data_folder"
log_dir="./results/colon_test_"$roi_x"_"$roi_y"_"$roi_z"_"$v_min"_"$v_max"_$data_folder"


	python ./swincell/train_main.py --data_dir=$data_dir --val_every=100 --distributed \
 --a_min=$v_min --a_max=$v_max --logdir $log_dir --max_epochs 5000 \
--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  --use_checkpoint --feature_size=48 --save_checkpoint --use_flows 





