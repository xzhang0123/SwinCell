import os
# import pdb
# import shutil
import glob
from natsort import natsorted
import pandas as pd
import time
import argparse
import numpy as np
import torch
import tifffile
# from tensorboardX import SummaryWriter

from swincell.utils.utils import batch_matching
from swincell.cellpose_dynamics import compute_masks
from functools import partial


parser = argparse.ArgumentParser(description="SwinCell Inference")
# data file in a list of data folder
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--gt_folder', type=str, help='ground truth folder.')
group.add_argument('--prediction_folder', type=str, help='prediction folder containing predicted masks.')
parser.add_argument("--output_dir", default='./results/eval.csv', help="folder to save results")
parser.add_argument("--output_file", default='eval.csv', help="folder to save results")
parser.add_argument("--matching_thresholds", default=[0.5,0.6,0.7,0.8,0.9,1], help="matching thresholds for evaluation") 


parser.add_argument("--downsample_factor", default=1, type=int, help="downsampling rate of input data, increase it when input images have very high resolution")

# parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")


def main_evaluation():

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    gt_file_list = natsorted(glob.glob(args.gt_folder + '/*.tif*'))
    pred_file_list = natsorted(glob.glob(args.prediction_folder + '/*.tif*'))
    df_match = batch_matching(gt_file_list, pred_file_list, args.matching_thresholds, args.output_dir, args.downsample_factor)
    df_match.to_csv(args.output_dir + args.output_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)






if __name__ == "__main__":
    main_evaluation()
