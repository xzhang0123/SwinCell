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
from monai.networks.nets import SwinUNETR, UNet
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from swincell.utils.utils import AverageMeter, distributed_all_gather
from swincell.utils.data_utils import folder_loader
from swincell.cellpose_dynamics import compute_masks
from functools import partial


parser = argparse.ArgumentParser(description="SwinCell Inference")
# data file in a list of data folder
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--data_file_list", default='', help="data file names in a list")
group.add_argument('--data_folder', type=str, help='folder containing data.')
parser.add_argument("--output_dir", default='./results', help="folder to save results")
parser.add_argument("--model_dir", default='./results/model.pt', help="path to saved model")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels, #cell probability channel + #flow channels")
parser.add_argument("--a_min", default=0, type=float, help="cliped min input value")
parser.add_argument("--a_max", default=255, type=float, help="cliped max input value")
parser.add_argument("--b_min", default=0.0, type=float, help="min target (output) value")
parser.add_argument("--b_max", default=1.0, type=float, help="max target value")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--space_x", default=1, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--downsample_factor", default=1, type=int, help="downsampling rate of input data, increase it when input images have very high resolution")
# parser.add_argument("--logdir", default="./results/test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--model", default="swin", type=str, help="Model Architecture")
# parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
# parser.add_argument("--data_dir", default="/data/", type=str, help="dataset directory")
parser.add_argument("--dataset", default="colon", type=str, help="dataset name")
# parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")


def main_infer():
    args = parser.parse_args()
    args.test_mode = True
    torch.cuda.set_device(args.gpu)
    if args.data_folder:
        infer_loader = folder_loader(args)
    else:
        infer_loader = None

    if args.model =='unet':  # used for ablation study
        model = UNet(
            # img_size=(args.roi_x, args.roi_y, args.roi_z),
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2),
            

        )
    elif args.model =='swin':
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,

        )
    else:
        raise Exception("Model not defined")  
    model_dict = torch.load(args.model_dir)["state_dict"]
    model.load_state_dict(model_dict)

    infer_step(model, infer_loader, args)
def infer_step(model, loader, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    """
    Perform inference step on the given model using the provided data loader.
    
    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (torch.utils.data.DataLoader): The data loader containing the input data.
        args (Namespace): The command-line arguments.
        model_inferer (callable, optional): The function to use for sliding window inference. Defaults to None.
        post_sigmoid (callable, optional): The function to apply after applying sigmoid activation. Defaults to None.
        post_pred (callable, optional): The function to apply to the predicted values. Defaults to None.
    
    Returns:
        None
    """
 
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





if __name__ == "__main__":
    main_infer()
