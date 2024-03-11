import numpy as np
import logging
import sys
from typing import List
from scipy.ndimage import zoom
import os
from scipy import ndimage as ndi
from scipy import stats
import argparse
from matplotlib import pyplot as plt

import yaml

def load_config(config_path):
    import torch
    config = _load_config_yaml(config_path)
    # Get a device to train on
    device_name = config.get('device', 'cuda:0')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.full_load(open(config_file, 'r'))

def get_samplers(num_training_data, validation_ratio, my_seed):
    from torch.utils.data import sampler as torch_sampler
    indices = list(range(num_training_data))
    split = int(np.floor(validation_ratio * num_training_data))

    np.random.seed(my_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = torch_sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch_sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler

def simple_norm(img, a, b, m_high=-1, m_low=-1):
    idx = np.ones(img.shape, dtype=bool)
    if m_high>0:
        idx = np.logical_and(idx, img<m_high)
    if m_low>0:
        idx = np.logical_and(idx, img>m_low)
    img_valid = img[idx]
    m,s = stats.norm.fit(img_valid.flat)
    strech_min = max(m - a*s, img.min())
    strech_max = min(m + b*s, img.max())
    img[img>strech_max]=strech_max
    img[img<strech_min]=strech_min
    img = (img- strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
    return img

def background_sub(img, r):
    struct_img_smooth = ndi.gaussian_filter(img, sigma=r, mode='nearest', truncate=3.0)
    struct_img_smooth_sub = img - struct_img_smooth
    struct_img = (struct_img_smooth_sub - struct_img_smooth_sub.min())/(struct_img_smooth_sub.max()-struct_img_smooth_sub.min())
    return struct_img

def get_random_cmap(num, seed=1, background=1):
    """
    Generate a random cmap

    Parameters
    ----------
    num : int
        The number of colors to be generated
    seed : int
        The same value will lead to the same cmap
    BACKGROUND : int
        The color of the background
    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The generated cmap
    """
    vals = np.linspace(0, 1, num + 1)
    np.random.seed(seed)
    np.random.shuffle(vals)
    vals = np.concatenate(([0], vals[1:]))
    cmap = plt.cm.colors.ListedColormap(plt.cm.rainbow(vals))
    cmap.colors[0, :3] = background
    return cmap

def volume_render_color(input_img,a=0.01,seed = 1):
    import cv2
    res_image = np.zeros((input_img.shape[0],input_img.shape[1],3))
    nor = input_img.max()
    for i in range(input_img.shape[2]):

        cur_img = input_img[:,:,i].astype('uint8')
        cur_img_c = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)
        
        cur_img_c1 = cv2.LUT(cur_img_c.astype(np.uint8),build_lut_cv2(seed=seed))
        # r,g,b,a = transferFunction(np.log(cur_img))
        r =cur_img_c1[:,:,0]
        g =cur_img_c1[:,:,1]
        b =cur_img_c1[:,:,2]
        res_image[:,:,0] = a*r + (1-a)*res_image[:,:,0]
        res_image[:,:,1] = a*g + (1-a)*res_image[:,:,1]
        res_image[:,:,2] = a*b + (1-a)*res_image[:,:,2]
    res_image = res_image.astype('float')
    res_image = np.clip((res_image/nor),0.0,1.0).astype('float')
    return res_image


def build_lut_cv2(num=256,seed=1):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)


    lut[:, 0, 0] =np.linspace(0, 256, 256)
    lut[:, 0, 1] =np.linspace(0, 256, 256)
    lut[:, 0, 2] = np.linspace(0, 256, 256)
    np.random.seed(seed)
    np.random.shuffle(lut[:, 0, 0])
    np.random.shuffle(lut[:, 0, 1])
    np.random.shuffle(lut[:, 0, 2])
    # lut[200:,0,:]=0
    lut[:1,0,:]=1
    
    return lut


def compute_iou(prediction, gt, cmap):

    area_i = np.logical_and(prediction, gt)
    area_i[cmap==0]=False
    area_u = np.logical_or(prediction, gt)
    area_u[cmap==0]=False

    return np.count_nonzero(area_i) / np.count_nonzero(area_u)

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger