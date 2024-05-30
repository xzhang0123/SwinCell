#!/bin/bash

#settings
root_folder='colon_dataset'
data_folder="/colon_10_no_tag"
roi_x=128   
roi_y=128
roi_z=32
v_min=80
v_max=255

data_dir="$root_folder""/$data_folder"
output_dir="./results/infer_colon_test_N5_"$roi_x"_"$roi_y"_"$roi_z"_"$v_min"_"$v_max"_$data_folder"


	python ./swincell/inference.py --data_folder=$data_dir --output_dir $output_dir \
 --a_min=$v_min --a_max=$v_max \
--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  





