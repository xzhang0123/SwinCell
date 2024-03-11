


#output_path=""
#data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_augmented"
data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_1_99"

#pre_trained_model='/home/xzhang/Projects/cellpose/pretrained_models/nanolive_v6_swinunet_cellpose_nor.pt'
pre_trained_model=''
amin=1
amax=100
#amin=15000
#amx=31000
roix=64
roiy=64
for dsp_rate in {1,2,3,4,5}
do
log_dir="Nanolive_mem_cellpose_unet_${roix}_${roiy}_32_v6_nor_dsp${dsp_rate}_test_contrast_scratch_no_aug_5fold"
	python finetune_main_cellpose.py --json_list='' --data_dir=$data_dir --val_every=100 --noamp --distributed --model 'swinunetr' --fold 1 \
 --pretrained_model_name=$pre_trained_model --a_min=$amin --a_max=$amax --logdir $log_dir --max_epochs 5000 --dsp $dsp_rate \
--roi_x=$roix --roi_y=$roiy --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained
sleep 200
done


#output_path=""
#data_dir="/data/download_data/quilt-data-access-tutorials-main/all_fov/allen100"
#log_dir="allen_cellpose100_unet_128_128_32_v1"
#	python finetune_main_cellpose.py --json_list='' --data_dir=$data_dir --val_every=50 --noamp --distributed --model 'unet' \
#--pretrained_model_name '' --a_min=385 --a_max=481 --logdir $log_dir --max_epochs 5000 \
#--roi_x=128 --roi_y=128 --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained



#data_dir="/data/download_data/quilt-data-access-tutorials-main/all_fov/allen100"
#log_dir="allen_cellpose100_unet_lr2e4_128_128_32_v2"
#log_dir="allen_cellpose100_swinunetr_lr2e4_128_128_32_v1"
#log_dir="allen_cellpose100_swinunetr_lr1e3_128_128_32_v1"
#	python finetune_main_cellpose.py --json_list='' --data_dir=$data_dir --val_every=100 --noamp --distributed --model 'unet' \
#--pretrained_model_name '/home/xzhang/Projects/cellpose/pretrained_models/allen_unet_v1.pt' --a_min=385 --a_max=481 --logdir $log_dir --max_epochs 5000 --optim_lr 2e-4 \
#--roi_x=128 --roi_y=128 --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
