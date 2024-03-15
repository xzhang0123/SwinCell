# SwinCell: a transformer-based framework for dense 3D cellular segmentation 


## Requiments
see requiments.txt
## Geting Started
### Create the environment:
```bash
conda create --name swincell numpy==1.21.5
conda activate swincell
```
```bash
cd <swincell_folder>
pip install .
```
for developers, install with
```bash
pip install -e .
```


<!-- ### Install with pip
```bash
pip install swincell
``` -->
## Data directory layout
    .
    ├── swincell
    ├── data_root_folder                    
    │   ├── images         # raw images in tiff format
    │   └── labels         # ground truth semantic label. 0=background, 1=cell
    └── ...
## Model training
### Model training with jupyter-notebook
[workflow_training](https://github.com/xzhang0123/SwinCell/blob/main/swincell/notebooks/workflow.ipynb)
## Model training via Terminal
```bash
# activate environment
conda acivate swincell
# configure hyper-paramters, run training
sh python ./swincell/train_main.py --data_dir=<data_dir> --val_every=<valid_every_N_ephochs> --model 'swin'  --logdir <log_dir> --max_epochs 100 --roi_x=64 --roi_y=64 --roi_z=32  --feature_size=48 \
```
## Model inference
TO DO 


```bash
#trouble shooting/common mistakes:
#feature size of swin-transformer must be divisible by 32, error raised otherwise
#final ROI size (patch size) must be higher than input image after downsampling (if dsp>1)