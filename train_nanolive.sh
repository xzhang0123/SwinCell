

root_dir=""
data_dir="./Nanolive_dataset/"
roi_x=128   
roi_y=128
roi_z=32

v_min=1
v_max=255

log_dir="./results/nanolive_"$roi_x"_"$roi_y"_"$roi_z"_"$v_min"_"$v_max"_$data_folder"

pre_trained_model=''


for fold in {1..5}
do
	python ./swincell/train_main.py --data_dir=$data_dir --val_every=100 --distributed --model 'swin' --a_min=$v_min --a_max=$v_max --logdir $log_dir --max_epochs 5000  --fold $fold \
--roi_x=128 --roi_y=128 --roi_z=32  --use_checkpoint --feature_size=48 --save_checkpoint --use_flows \

sleep 200
done



