


#output_path=""
#data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_augmented"
data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_1_99"
log_dir="Nanolive_mem_cellpose_unet_128_128_32_v6_nor_dsp1_unet_ablation_study"
#pre_trained_model='/home/xzhang/Projects/cellpose/pretrained_models/nanolive_v6_swinunet_cellpose_nor.pt'
pre_trained_model=''
amin=1
amax=100
#amin=15000
#amx=31000

for fold in {1..5}
do
	python ./swincell/train_main.py --data_dir=$data_dir --val_every=100 --distributed --model 'unet' \
 --pretrained_model_name=$pre_trained_model --a_min=$amin --a_max=$amax --logdir $log_dir --max_epochs 10000 --dsp 1 --fold $fold \
--roi_x=128 --roi_y=128 --roi_z=32  --in_channels=1 --out_channels=1 --spatial_dims=3 --use_checkpoint --feature_size=48 --save_checkpoint  \
#--use_ssl_pretrained
sleep 200
done



