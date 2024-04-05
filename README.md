# SwinCell: a transformer-based framework for dense 3D cellular segmentation 


## Requiments
see requiments.txt
## Getting Started
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
## Demo data
[Nanolive Demo Dataset](https://brookhavenlab-my.sharepoint.com/:f:/g/personal/xzhang4_bnl_gov/EsDdL48uEmRKskKE5OCOX4cBaOXSdmS-YGWDxlS7_lgExA?e=WyDpCh)
## Model training
### Model training with jupyter-notebook
1. add the SwinCell environment as a new kernel to your Jupyter Notebook: 
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=swincell
```
2. Run the following notebook under the swincell environment:
[demo_workflows.ipynb](https://github.com/xzhang0123/SwinCell/blob/main/swincell/notebooks/training_and_prediction_pipeline.ipynb)

A google colab notebook will be added soon.
### Model training via Terminal
```bash
# activate environment
conda acivate swincell
# configure hyper-paramters, run training
sh python ./swincell/train_main.py --data_dir=<data_dir> --val_every=<valid_every_N_ephochs> 
--model 'swin'  --logdir <log_dir> --max_epoches <max_epoches> --roi_x=<roi_x> --roi_y=<roi_y> --roi_z=<roi_z>

```
## Model inference
[model_inference](https://github.com/xzhang0123/SwinCell/blob/main/swincell/notebooks/)

## Trouble shooting/common mistakes:
* feature size of swin-transformer must be divisible by 32, error raises otherwise
* final ROI size (patch size) must be larger than input image after downsampling (if dsp>1)