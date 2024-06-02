#!/bin/bash
#predict with different models, and compare their performance
#settings
root_folder='colon_dataset'
#data_folder="/colon_10_no_tag"
data_folder="/colon_last5_no_tag_predict"
roi_x=256  
roi_y=256
roi_z=32
v_min=1
v_max=255

data_dir="$root_folder""/$data_folder"

output_dir=$data_dir'/colon_test_N5_128_128_32_1_255_colon_25_predict'
model_path='./results/colon_test_N5_128_128_32_1_255_colon_25/model_final.pt'

	python ./swincell/inference.py --data_folder=$data_dir --model_dir $model_path --output_dir $output_dir \
 --a_min=$v_min --a_max=$v_max \
--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  

exit
#sleep 300

output_dir=$data_dir'/colon_test_N5_128_128_32_1_255_colon_10_no_tag_predict_infervmin1'
model_path='./results/colon_test_N5_128_128_32_1_255_/colon_10_no_tag/model_final.pt'
	python ./swincell/inference.py --data_folder=$data_dir --model_dir $model_path --output_dir $output_dir \
 --a_min=$v_min --a_max=$v_max \
--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  


sleep 300

output_dir=$data_dir'/colon_test_N5_128_128_32_1_255_colon_5_no_tag_predict_infervmin1'
model_path='./results/colon_test_N5_128_128_32_1_255_/colon_5/model_final.pt'
	python ./swincell/inference.py --data_folder=$data_dir --model_dir $model_path --output_dir $output_dir \
 --a_min=$v_min --a_max=$v_max \
--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  

sleep 300

output_dir=$data_dir'/colon_test_N5_128_128_32_1_255_colon_20_no_tag_predict_infervmin1'
model_path='./results/colon_test_N5_128_128_32_1_255_/colon_20/model_final.pt'
	python ./swincell/inference.py --data_folder=$data_dir --model_dir $model_path --output_dir $output_dir \
 --a_min=$v_min --a_max=$v_max \
--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  


#output_dir=$data_dir'/colon_test_N5_128_128_32_80_255_colon_10_no_tag_predict'
#model_path='./results/colon_test_N5_128_128_32_80_255_colon_10_no_tag/model_final.pt'
#echo $output_dir
#echo $model_path 
#	python ./swincell/inference.py --data_folder=$data_dir --model_dir $model_path --output_dir $output_dir \
# --a_min=$v_min --a_max=$v_max \
#--roi_x=$roi_x --roi_y=$roi_y --roi_z=$roi_z  

#sleep 300
