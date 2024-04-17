import os
# import pdb
# import shutil
import time
import argparse
import numpy as np
import torch
# import torch.nn.parallel
# import torch.utils.data.distributed
import tifffile
# from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast
from swincell.utils.utils import AverageMeter, distributed_all_gather
from swincell.cellpose_dynamics import compute_masks
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from functools import partial


parser = argparse.ArgumentParser(description="SwinCell Training")
parser.add_argument("--output_dir", default='./results', help="start training from saved checkpoint")
# parser.add_argument("--logdir", default="./results/test", type=str, help="directory to save the tensorboard logs")
# parser.add_argument("--model", default="swin", type=str, help="Model Architecture")
# parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
# parser.add_argument("--data_dir", default="/data/", type=str, help="dataset directory")
parser.add_argument("--dataset", default="colon", type=str, help="dataset name")
# parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")


def infer_step(model, loader, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    if not model_inferer:
        infer_ROI = (256,256,32)
        model_inferer = partial(
        sliding_window_inference,
        roi_size=infer_ROI,
        sw_batch_size=2,
        predictor=model,
        overlap=0.5,
        mode='gaussian'
    )
    model.eval()


    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"]
            output_filename = str(idx) + '_prediction.tiff'
            logits = model_inferer(data)
            logits_out =  np.squeeze(logits.detach().cpu().numpy())
            # cell probs channel
            logits_out[0] = post_pred(post_sigmoid(logits_out[0]))
            #validate with the binary masks 
            logits_out_transposed = np.transpose(logits_out,(0,3,2,1))
            acc = acc.cuda(args.rank)

            masks_recon,p = compute_masks(logits_out_transposed[1:4,:,:,:],logits_out_transposed[0,:,:,:],cellprob_threshold=0.4,flow_threshold=0.4, do_3D=True,min_size=2500, use_gpu=True)

                
            # tifffile.imwrite(output_folder +'/test_logits_'+out_filename ,logits_out)
            # tifffile.imwrite(output_folder +'/test_logits_transposed,'+out_filename ,logits_out_transposed)
            tifffile.imwrite(args.output_folder +'/masks_'+output_filename ,masks_recon)