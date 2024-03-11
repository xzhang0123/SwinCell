#!/bin/bash


#settings
data_folder="colon_30"
#model_name='unet'
model_name='swinunetr'
data_dir="/data/download_data/colon_dataset/$data_folder"

log_dir="colon_cellpose_"$model_name"_128_128_32_v2_1_255_$data_folder"


	python finetune_main_cellpose.py --json_list='' --data_dir=$data_dir --val_every=100 --noamp --distributed --model $model_name \
 --pretrained_model_name '/home/xzhang/Projects/cellpose/pretrained_models/colon15_swinunetr_v1.pt' --a_min=1 --a_max=255 --logdir $log_dir --max_epochs 10000 \
--roi_x=128 --roi_y=128 --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained





