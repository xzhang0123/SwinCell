#!/bin/bash


#settings
data_folder="colon_5"
#model_name='unet'
model_name='swin'
data_dir="/data/download_data/colon_dataset/$data_folder"

log_dir="./results/colon_test_tutorial_"$model_name"_$data_folder"


	python ./swincell/train_main.py --data_dir=$data_dir --val_every=2 --noamp --distributed --model $model_name \
--a_min=1 --a_max=255 --logdir $log_dir --max_epochs 10 --dataset 'colon' --dsp=1 \
--roi_x=32 --roi_y=32 --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained





