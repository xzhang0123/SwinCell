


#output_path=""
#data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_augmented"
data_dir="/data/download_data/Nanolive_mem_cellpose_data_v5_nor_1_99"
log_dir="test_nanolive"
#pre_trained_model='/home/xzhang/Projects/cellpose/pretrained_models/nanolive_v6_swinunet_cellpose_nor.pt'
pre_trained_model=''
amin=1
amax=100
fold=1
#amin=15000
#amx=31000

	python ./swincell/train_main.py --json_list='' --data_dir=$data_dir --val_every=1 --noamp --distributed --model 'swin' --dataset 'nanolive' \
 --pretrained_model_name=$pre_trained_model --a_min=$amin --a_max=$amax --logdir $log_dir --max_epochs 5 --dsp 1 --fold $fold \
--roi_x=64 --roi_y=64 --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained



