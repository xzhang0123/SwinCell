#!/bin/bash


#settings
data_folder="colon_30"
#model_name='unet'
model_name='swinunetr'
data_dir="/data/download_data/colon_dataset/$data_folder"

log_dir="colon_test_tutorial_"$model_name"_128_128_32_v2_1_255_$data_folder"


	python train_main.py --data_dir=$data_dir --val_every=2 --noamp --distributed --model $model_name \
--a_min=1 --a_max=255 --logdir $log_dir --max_epochs 10 \
--roi_x=32 --roi_y=32 --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained





