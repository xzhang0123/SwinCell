# SwinCell: a transformer-based framework for dense 3D cellular segmentation 
A 3D transformer-based framework [1] that leverages the Swin-Transformer architecture for flow prediction [2], enabling efficient segmentation of individual cell instances in 3D

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
## Nanolive Demo data
[Nanolive Demo Dataset](https://brookhavenlab-my.sharepoint.com/:f:/g/personal/xzhang4_bnl_gov/EsDdL48uEmRKskKE5OCOX4cBaOXSdmS-YGWDxlS7_lgExA?e=WyDpCh)
## Link to the Colon dataset
[Colon Dataset](http://datasets.gryf.fi.muni.cz/iciar2011/ColonTissue_LowNoise_3D_TIFF.zip)

Note: The original colon dataset contains a private TIFF tag 65000 (0xFDE8) that is not recognized by standard TIFF reading libraries. To prevent continuous Warnings during model training, we provide a cleaned demo version of the dataset. You can download the updated dataset from the link below

[Cleaned Colon Dataset](https://brookhavenlab-my.sharepoint.com/:u:/g/personal/xzhang4_bnl_gov/EaNWJnxUgYVFgzpE_du_VrEBUgJ-jyssLkklff3Ii8jZ8g?e=RONfch)
## Model training
### Model training with jupyter-notebook
1. add the SwinCell environment as a new kernel to your Jupyter Notebook: 
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=swincell
```
2. Run the following demo notebook under the swincell environment:
[training_prediction_pipeline.ipynb](https://github.com/xzhang0123/SwinCell/blob/main/swincell/notebooks/training_prediction_pipeline.ipynb)

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
[training_prediction_pipeline.ipynb](https://github.com/xzhang0123/SwinCell/blob/main/swincell/notebooks/training_prediction_pipeline.ipynb)

## Trouble shooting/common mistakes:
* feature size of swin-transformer must be divisible by 32, error raises otherwise
* final ROI size (patch size) must be larger than input image after downsampling (if dsp>1)

## References

1.  Zhang, X. et al. SwinCell: a transformer-based framework for dense 3D cellular segmentation. 2024.04.05.588365 Preprint at https://doi.org/10.1101/2024.04.05.588365 (2024).
2.	Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. Nat. Methods 18, 100–106 (2021).
