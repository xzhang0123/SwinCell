


#output_path=""
data_dir="/home/xzhang/Projects/cellpose/Nanolive_mem_cellpose_data_v5_nor_1_99"
log_dir="Nanolive_mem_cellpose_unet_128_128_32_v6_nor_dsp1_unet_ablation_study"
pre_trained_model=''


for fold in {1..5}
do
	python ./swincell/train_main.py --data_dir=$data_dir --val_every=100 --distributed --model 'unet' \ --a_min=1 --a_max=255 --logdir $log_dir --max_epochs 10000 --dsp 1 --fold $fold \
--roi_x=128 --roi_y=128 --roi_z=32  --use_checkpoint --feature_size=48 --save_checkpoint  \
#--use_ssl_pretrained
sleep 200
done



