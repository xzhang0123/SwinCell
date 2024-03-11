
import numpy as np
import torch
from matplotlib import pyplot as plt


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
def plot_with_se(ax,matrix,iou_thresholds,label=None,style='-',color=None):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    se = std / np.sqrt(matrix.shape[0])
    ax.plot(iou_thresholds,mean,linewidth=2,linestyle=style,color=color)

    ax.fill_between(iou_thresholds,mean-se,mean+se,alpha=0.3,label=label,color=color,edgecolor=None)
    return ax

def plot_box_with_violin(ax,data,label=None,style='-',facecolorlist=None):
    N=len(facecolorlist)
    box_plot = ax.boxplot(data)
    violin_plot = ax.violinplot(data, showextrema=False)
    for i in box_plot['medians']:
        i.set_color('black')  # Set the color to black
    for i,pc in enumerate(violin_plot['bodies']):
        color = facecolorlist[i%N]
        pc.set_facecolor(color)  # Fill color of the violin plot

    return ax


# def calculate_cell_volumes(mask):
#     # Get unique labels excluding the background label (0)
#     labels = np.unique(mask)
#     labels = labels[labels != 0]

#     # Calculate the volume of each cell
#     volumes = []
#     for label in labels:
#         cell_volume = np.sum(mask == label)
#         volumes.append(cell_volume)

#     return volumes

# def calculate_cell_diameters(mask):
#     from scipy import ndimage
#     # Calculate the Euclidean distance transform of the mask
#     distance_transform = ndimage.distance_transform_edt(mask)
    
#     # Initialize an empty list to store diameters
#     cell_diameters = []
    
#     # Iterate through each label (cell)
#     for label in range(1, mask.max() + 1):
#         # Extract the region of the current cell
#         cell_region = (mask == label)
        
#         # Calculate the maximum distance within the cell region
#         max_distance = np.max(distance_transform[cell_region])
        
#         # Calculate the diameter as twice the maximum distance
#         diameter = 2 * max_distance
        
#         # Append the diameter to the list
#         cell_diameters.append(diameter)
    
#     return cell_diameters

# def dice(x, y):
#     intersect = np.sum(np.sum(np.sum(x * y)))
#     y_sum = np.sum(np.sum(np.sum(y)))
#     if y_sum == 0:
#         return 0.0
#     x_sum = np.sum(np.sum(np.sum(x)))
#     return 2 * intersect / (x_sum + y_sum)


# class AverageMeter(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


# def distributed_all_gather(
#     tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
# ):

#     if world_size is None:
#         world_size = torch.distributed.get_world_size()
#     if valid_batch_size is not None:
#         valid_batch_size = min(valid_batch_size, world_size)
#     elif is_valid is not None:
#         is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
#     if not no_barrier:
#         torch.distributed.barrier()
#     tensor_list_out = []
#     with torch.no_grad():
#         if is_valid is not None:
#             is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
#             torch.distributed.all_gather(is_valid_list, is_valid)
#             is_valid = [x.item() for x in is_valid_list]
#         for tensor in tensor_list:
#             gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
#             torch.distributed.all_gather(gather_list, tensor)
#             if valid_batch_size is not None:
#                 gather_list = gather_list[:valid_batch_size]
#             elif is_valid is not None:
#                 gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
#             if out_numpy:
#                 gather_list = [t.cpu().numpy() for t in gather_list]
#             tensor_list_out.append(gather_list)
#     return tensor_list_out
