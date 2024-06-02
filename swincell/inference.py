import os
# import pdb
# import shutil
import time
import argparse
import numpy as np
import torch
import tifffile
# from tensorboardX import SummaryWriter
from monai.networks.nets import SwinUNETR, UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from swincell.utils.utils import load_model
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
parser.add_argument("--gpu", default=1, type=int, help="GPU id to use")
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
parser.add_argument("--workers", default=2, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--downsample_factor", default=1, type=int, help="downsampling rate of input data, increase it when input images have very high resolution")
# parser.add_argument("--logdir", default="./results/test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--model", default="swin", type=str, help="Model Architecture")
# parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
# parser.add_argument("--data_dir", default="/data/", type=str, help="dataset directory")
parser.add_argument("--dataset", default="colon", type=str, help="dataset name")
parser.add_argument("--save_flows", action="store_true", help="save predicted flows")

# parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")


def main_infer():
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
        Non
    """
    args = parser.parse_args()
    args.test_mode = True
    args.checkpoint = False
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.data_folder:
        infer_loader = folder_loader(args)
    else:
        infer_loader = None

    model = load_model(args).to(device)
    model_dict = torch.load(args.model_dir)["state_dict"]
    model.load_state_dict(model_dict)
 
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=2,
        predictor=model,
        overlap=args.infer_overlap,
        mode='gaussian'
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with torch.no_grad():
        for idx, batch_data in enumerate(infer_loader):
            data = batch_data["image"].to(device)
            output_filename = str(idx) + '_prediction.tiff'
            print(data.shape,)
            print('predicting', output_filename)
            # print(model)
            # print(data)
            logits = model_inferer(data)
            logits_out =  np.squeeze(logits.detach().cpu().numpy())
            # cell probs channel
            print('logits_out shape', logits_out.shape)
            logits_out[0] = post_pred(post_sigmoid(logits_out[0]))
            #validate with the binary masks 
            logits_out_transposed = np.transpose(logits_out,(0,3,2,1))
            # acc = acc.cuda(args.rank)
            print('logits_out_transposed shape', logits_out_transposed.shape)
            masks_recon,p = compute_masks(logits_out_transposed[1:4,:,:,:],logits_out_transposed[0,:,:,:],cellprob_threshold=0.4,flow_threshold=0.4, do_3D=True,min_size=2500, use_gpu=True)

            if args.save_flows:
                tifffile.imwrite(args.output_dir +'/logits_transposed,'+output_filename ,logits_out_transposed)
            # tifffile.imwrite(output_dir +'/test_logits_'+out_filename ,logits_out)
            # tifffile.imwrite(output_dir +'/test_logits_transposed,'+out_filename ,logits_out_transposed)
            tifffile.imwrite(args.output_dir +'/masks_'+output_filename ,masks_recon)





if __name__ == "__main__":
    main_infer()
