# training with different downsampling rate


#output_path=""
data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_1_99"
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
	python ./swincell/train_main.py  --data_dir=$data_dir --val_every=100 --distributed --model 'swin' --fold 1 \
 --pretrained_model_name=$pre_trained_model --a_min=$amin --a_max=$amax --logdir $log_dir --max_epochs 5000 --dsp $dsp_rate \
--roi_x=$roix --roi_y=$roiy --roi_z=32  --in_channels=1 --out_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint --cellpose \
#--use_ssl_pretrained
sleep 200
done

