
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import SwinUNETR,UNet
import tifffile

def load_default_config():
    """
    Load the default configuration for the model training process.
    Returns:
        SimpleNamespace: An object with the following attributes set to default values:
            - data_dir (str): Directory path for the data.
            - dataset (str): Name of the dataset, used to define dataset specific parameters.
            - checkpoint (str): Path to the model checkpoint.
            - rank (int): Rank of the process for distributed training.
            - batch_size (int): Number of samples per batch, default is 1.
            - max_epochs (int): Maximum number of training epochs, default is 1000.
            - optim_lr (float): Learning rate for the optimizer, default is 1e-5.
            - model (str): Model type to use, default is 'swin'.
            - distributed (bool): Flag to enable distributed training, default is False.
            - optimizer (str): Type of optimizer to use, default is 'adam'.
            - save_temp_img (bool): Flag to enable saving temporary prediction images, used for debugging purposes.
            - downsample_factor (int): Factor to downsample the input images, default is 1.
            - use_flows (bool): Flag to determine if flows should be used, default is True.
            - roi_x (int), roi_y (int), roi_z (int): Dimensions of the region of interest,
              defaults are 128, 128, and 32 respectively.
            - fold (str): Specific fold of data to use, default is None.
            - workers (int): Number of worker threads for data loading, default is 8.
    """
    from types import SimpleNamespace
    args = SimpleNamespace(
    data_dir =None,
    dataset='colon',
    checkpoint=None,
    rank=0,
    a_min=0,
    a_max=255,
    b_min=0,
    b_max=1,
    batch_size=1,
    max_epochs=1000,
    optim_lr=1e-5,
    model='swin',
    distributed=False,
    optimizer='adam',
    save_temp_img=False,
    downsample_factor =1, # downsample the input image if resolution is too high
    weight_decay=0.0001,
    feature_size=48,
    use_flows=True,
    roi_x=128,
    roi_y=128,
    roi_z=32,
    fold=None,
    workers=8,

)
    return args

def load_model(args):

    """
        Load and initialize a segmentation model based on the specified arguments.

        Returns:
            object: An instance of the specified segmentation model, configured and ready for training
            or inference.
        Raises:
            NotImplementedError: If the model specified in 'args.model' is not supported.

        Examples:
            >>> from types import SimpleNamespace
            >>> args = SimpleNamespace(model='swin', roi_x=128, roi_y=128, roi_z=32,
                                    feature_size=48, checkpoint=None)
            >>> model = load_model(args)
            >>> print(model)
    """
 
    if args.model == 'swin':
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=4,
            # feature_size=args.feature_size,
            feature_size=args.feature_size,
            use_checkpoint=args.checkpoint,
        )
    elif args.model == 'unet': # for ablation study
        model = UNet(
            # img_size=(args.roi_x, args.roi_y, args.roi_z),
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2
        )
    else:
        raise NotImplementedError 
    return model


def auto_gamma_correction(image):
    """
    Apply automatic gamma correction to the input image.
    
    Parameters:
    image (numpy.ndarray): Input image (gray scale).
    
    Returns:
    numpy.ndarray: Gamma corrected image.
    """

    meanv = np.mean(image)
    gamma = np.log(0.5) / np.log(meanv / 255.0 + 1e-8)
    return np.power(image / 255.0, gamma) * 255

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
    """
    Plot the mean and standard error of the evaluation metrics for a given IoU threshold, used to plot results in papers
    Returns:
    matplotlib.axes.Axes: The axis object with the plot.
    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    se = std / np.sqrt(matrix.shape[0])
    ax.plot(iou_thresholds,mean,linewidth=2,linestyle=style,color=color)

    ax.fill_between(iou_thresholds,mean-se,mean+se,alpha=0.3,label=label,color=color,edgecolor=None)
    return ax

def plot_with_std(ax,matrix,iou_thresholds,label=None,style='-',color=None):
    """
    Plot the mean and std of the evaluation metrics for a given IoU threshold, used to plot results in papers
    Returns:
    matplotlib.axes.Axes: The axis object with the plot.
    """
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    # se = std / np.sqrt(matrix.shape[0])
    ax.plot(iou_thresholds,mean,linewidth=2,linestyle=style,color=color)

    ax.fill_between(iou_thresholds,mean-std,mean+std,alpha=0.3,label=label,color=color,edgecolor=None)
    return ax
def plot_box_with_violin(ax,data,label=None,style='-',facecolorlist=None):
    """
    Plot both boxplot and violin plot together on a given axis for visual comparison of data distribution.

    Parameters:
        ax (matplotlib.axes.Axes): The matplotlib axis object on which the plots will be drawn.
        data (array-like): The input data for the plots. Should be a list or array where each element
                           corresponds to a different dataset.
        label (str, optional): The label for the plots, used for the legend. Default is None.
        style (str, optional): The line style for the boxplot, not typically used as boxplots do not
                               have line styles. Default is '-'.
        facecolorlist (list, optional): A list of colors for the violin plots. If None, a default color
                                        will be used. Default is None.

    Returns:
        matplotlib.axes.Axes: The axis object with the plot.
    """
    N=len(facecolorlist)
    box_plot = ax.boxplot(data)
    violin_plot = ax.violinplot(data, showextrema=False)
    for i in box_plot['medians']:
        i.set_color('black')  # Set the color to black
    for i,pc in enumerate(violin_plot['bodies']):
        color = facecolorlist[i%N]
        pc.set_facecolor(color)  # Fill color of the violin plot

    return ax


def calculate_cell_volumes(mask):
    """
    Calculate the volumes of individual cells labeled in a segmentation mask. 0 is the background

    Parameters:
        mask (numpy.ndarray): A segmentation mask array where each unique non-zero integer
                              represents a unique cell.
    Returns:
        list: A list of volumes corresponding to each unique cell label in the mask, sorted by
              the label values.

    Examples:
        >>> mask = np.array([[0, 1, 1], [0, 2, 2], [0, 2, 2]])
        >>> volumes = calculate_cell_volumes(mask)
        >>> print(volumes)
        [2, 3]
    """

    # Get unique labels excluding the background label (0)
    labels = np.unique(mask)
    labels = labels[labels != 0]

    # Calculate the volume of each cell
    volumes = []
    for label in labels:
        cell_volume = np.sum(mask == label)
        volumes.append(cell_volume)

    return volumes

def calculate_cell_diameters(mask):

    """
    Calculate the diameters of individual 3D cells labeled in a segmentation mask based on the
    Euclidean distance transform. 0 is the background, cells are labeled with 1,2,3,...

    Parameters:
        mask (numpy.ndarray): A segmentation mask array where each unique non-zero integer
                              represents a unique cell.

    Returns:
        list: A list of diameters corresponding to each unique cell label in the mask, sorted by
              the label values.

    Examples:
        >>> mask = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 2, 2]])
        >>> diameters = calculate_cell_diameters(mask)
        >>> print(diameters)
        [2.8284271247461903, 2.8284271247461903]  # Example output based on default calculation settings
    """
    from scipy import ndimage
    # Calculate the Euclidean distance transform of the mask
    distance_transform = ndimage.distance_transform_edt(mask)
    
    # Initialize an empty list to store diameters
    cell_diameters = []
    
    # Iterate through each label (cell)
    for label in range(1, mask.max() + 1):
        # Extract the region of the current cell
        cell_region = (mask == label)
        
        # Calculate the maximum distance within the cell region
        max_distance = np.max(distance_transform[cell_region])
        
        # Calculate the diameter as twice the maximum distance
        diameter = 2 * max_distance
        
        # Append the diameter to the list
        cell_diameters.append(diameter)
    
    return cell_diameters

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def get_predefined_colors():
    """
    Returns a list of predefined colors normalized to the range [0, 1].

    :return: A list of normalized RGB color tuples.
    """
    colors = [
        (207 / 255, 54 / 255, 112 / 255),
        (83 / 255, 153 / 255, 179 / 255),
        (112 / 255, 170 / 255, 87 / 255),
        (115 / 255, 81 / 255, 155 / 255),
        (235 / 255, 123 / 255, 46 / 255),
    ]
    return colors

def normalize_image_percentile(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X

def normalize(Y):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    # x01 = np.percentile(X, lower)
    # x99 = np.percentile(X, upper)
    X = (X - np.min(X)) / (np.max(X)- np.min(X))
    return X

def fill_small_holes_3d(masks, min_size=1000,bin_closing_structure=np.ones((5,5,3)).astype(int)):
    """
    Fills holes in masks (3D) and discards masks smaller than min_size.

    This function fills holes in each mask using scipy.ndimage.morphology.binary_fill_holes.
    It also removes masks that are smaller than the specified min_size.

    Parameters:
    masks (ndarray): Int, 3D array of labelled masks.
        0 represents no mask, while positive integers represent mask labels.
        The size can be [Lz x Ly x Lx].
    min_size (int, optional): Minimum number of pixels per mask.
        Masks smaller than min_size will be removed.
        Set to -1 to turn off this functionality. Default is 1000.
    bin_closing_structure (ndarray, optional): Structuring element for binary closing operation.
        Default is a 3x3x3 array of ones.

    Returns:
    ndarray: Int, 3D array of masks with holes filled and small masks removed.
        0 represents no mask, while positive integers represent mask labels.
        The size is [Lz x Ly x Lx].
    """
    from scipy.ndimage import find_objects, binary_fill_holes, binary_closing
    from scipy.ndimage import label
    masks = masks.copy()
    masks,num_features = label(masks)
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            # if min_size > 0 and npix < min_size:
            #     print('delete')
            #     masks[slc][msk] = 0
            if npix > 0:  
                # if bin_closing_structure is not None:
                #     for _ in range(1): # repeat 3 times
                #         msk = binary_closing(msk,bin_closing_structure)
                if msk.ndim==3:
                    
                    for k in range(msk.shape[0]):
  
                        msk[k] = binary_closing(msk[k],bin_closing_structure)
                        msk[k] = binary_fill_holes(msk[k])
                
                else:          
                    msk = binary_fill_holes(msk)
                # masks[slc][msk] = (j+1)
                masks[slc][msk] = (j+1)
                j+=1
    return masks


# MASK Matching algorithm used by stardist algorithm:


from numba import jit
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from collections import namedtuple
from csbdeep.utils import _raise
from scipy.ndimage import find_objects
import cv2

matching_criteria = dict()


def distance_to_boundary(masks):
    """ get distance to boundary of mask pixels
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    dist_to_bound: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('distance_to_boundary takes 2D or 3D array, not %dD array'%masks.ndim)
    dist_to_bound = np.zeros(masks.shape, np.float64)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            dist_to_bound[i] = distance_to_boundary(masks[i])
        return dist_to_bound
    else:
        slices = find_objects(masks)
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T  
                ypix, xpix = np.nonzero(mask)
                min_dist = ((ypix[:,np.newaxis] - pvr)**2 + 
                            (xpix[:,np.newaxis] - pvc)**2).min(axis=1)
                dist_to_bound[ypix + sr.start, xpix + sc.start] = min_dist
        return dist_to_bound

def masks_to_edges(masks, threshold=1.0):
    """ get edges of masks as a 0-1 array 
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    edges: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are edge pixels

    """
    dist_to_bound = distance_to_boundary(masks)
    edges = (dist_to_bound < threshold) * (masks > 0)
    return edges

def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels)-{0}) == set(range(1,1+labels.max()))


def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)


def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if len(y) == 0:
        return True
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def label_overlap(x, y, check=True):
    if check:
        _check_label_array(x,'x',True)
        _check_label_array(y,'y',True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)

@jit(nopython=True)
def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap


def _safe_divide(x,y, eps=1e-10):
    """computes a safe divide which returns 0 if y is zero"""
    if np.isscalar(x) and np.isscalar(y):
        return x/y if np.abs(y)>eps else 0.0
    else:
        out = np.zeros(np.broadcast(x,y).shape, np.float32)
        np.divide(x,y, out=out, where=np.abs(y)>eps)
        return out


def intersection_over_union(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, (n_pixels_pred + n_pixels_true - overlap))

matching_criteria['iou'] = intersection_over_union


def intersection_over_true(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pixels_true)

matching_criteria['iot'] = intersection_over_true


def intersection_over_pred(overlap):
    _check_label_array(overlap,'overlap')
    if np.sum(overlap) == 0:
        return overlap
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    return _safe_divide(overlap, n_pixels_pred)

matching_criteria['iop'] = intersection_over_pred


def precision(tp, fp, fn):
    """Compute precision (= TP / (TP + FP)) from the number of true positives, false positives and false negatives.

    If there are no true positives, return 0 as precision is undefined in that case.

    Args:
        tp (int): Number of true positives (TP)
        fp (int): Number of false positives (FP)
        fn (int): Number of false negatives (FN)

    Returns:
        float: Precision value
    """
    return tp/(tp+fp) if tp > 0 else 0.0

def recall(tp,fp,fn):
    return tp/(tp+fn) if tp > 0 else 0
def accuracy(tp,fp,fn):
    # also known as "average precision" (?)
    # -> https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    return tp/(tp+fp+fn) if tp > 0 else 0
def f1(tp,fp,fn):
    # also known as "dice coefficient"
    return (2*tp)/(2*tp+fp+fn) if tp > 0 else 0


def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images from stardist code.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false negatives (fn)
    whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default IoU)
    report_matches: bool
        if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')

    Returns
    -------
    Matching object with different metrics as attributes

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true,5,axis = 0)

    >>> stats = matching(y_true, y_pred)
    >>> print(stats)
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0, recall=0, accuracy=0, f1=0, n_true=1, n_pred=1, mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

    """
    _check_label_array(y_true,'y_true')
    _check_label_array(y_pred,'y_pred')
    y_true.shape == y_pred.shape or _raise(ValueError("y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(y_true=y_true, y_pred=y_pred)))
    criterion in matching_criteria or _raise(ValueError("Matching criterion '%s' not supported." % criterion))
    if thresh is None: thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float,thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # ignoring background
    scores = scores[1:,1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        # not_trivial = n_matched > 0 and np.any(scores >= thr)
        not_trivial = n_matched > 0
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2*n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind,pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        fp = n_pred - tp
        fn = n_true - tp
        # assert tp+fp == n_pred
        # assert tp+fn == n_true

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind,pred_ind][match_ok]) if not_trivial else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score    = _safe_divide(sum_matched_score, n_true)
        panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

        stats_dict = dict (
            criterion          = criterion,
            thresh             = thr,
            fp                 = fp,
            tp                 = tp,
            fn                 = fn,
            precision          = precision(tp,fp,fn),
            recall             = recall(tp,fp,fn),
            accuracy           = accuracy(tp,fp,fn),
            f1                 = f1(tp,fp,fn),
            n_true             = n_true,
            n_pred             = n_pred,
            mean_true_score    = mean_true_score,
            mean_matched_score = mean_matched_score,
            panoptic_quality   = panoptic_quality,
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update (
                    # int() to be json serializable
                    matched_pairs  = tuple((int(map_rev_true[i]),int(map_rev_pred[j])) for i,j in zip(1+true_ind,1+pred_ind)),
                    matched_scores = tuple(scores[true_ind,pred_ind]),
                    matched_tps    = tuple(map(int,np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update (
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                )
        return namedtuple('Matching',stats_dict.keys())(*stats_dict.values())

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single,thresh))



def matching_dataset(y_true, y_pred, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):
    """matching metrics for list of images, see `stardist.matching.matching`
    """
    len(y_true) == len(y_pred) or _raise(ValueError("y_true and y_pred must have the same length."))
    return matching_dataset_lazy (
        tuple(zip(y_true,y_pred)), thresh=thresh, criterion=criterion, by_image=by_image, show_progress=show_progress, parallel=parallel,
    )



def matching_dataset_lazy(y_gen, thresh=0.5, criterion='iou', by_image=False, show_progress=True, parallel=False):

    expected_keys = set(('fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'))

    single_thresh = False
    if np.isscalar(thresh):
        single_thresh = True
        thresh = (thresh,)

    tqdm_kwargs = {}
    tqdm_kwargs['disable'] = not bool(show_progress)
    if int(show_progress) > 1:
        tqdm_kwargs['total'] = int(show_progress)

    # compute matching stats for every pair of label images
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        fn = lambda pair: matching(*pair, thresh=thresh, criterion=criterion, report_matches=False)
        with ThreadPoolExecutor() as pool:
            stats_all = tuple(pool.map(fn, tqdm(y_gen,**tqdm_kwargs)))
    else:
        stats_all = tuple (
            matching(y_t, y_p, thresh=thresh, criterion=criterion, report_matches=False)
            for y_t,y_p in tqdm(y_gen,**tqdm_kwargs)
        )

    # accumulate results over all images for each threshold separately
    n_images, n_threshs = len(stats_all), len(thresh)
    accumulate = [{} for _ in range(n_threshs)]
    for stats in stats_all:
        for i,s in enumerate(stats):
            acc = accumulate[i]
            for k,v in s._asdict().items():
                if k == 'mean_true_score' and not bool(by_image):
                    # convert mean_true_score to "sum_matched_score"
                    acc[k] = acc.setdefault(k,0) + v * s.n_true
                else:
                    try:
                        acc[k] = acc.setdefault(k,0) + v
                    except TypeError:
                        pass

    # normalize/compute 'precision', 'recall', 'accuracy', 'f1'
    for thr,acc in zip(thresh,accumulate):
        set(acc.keys()) == expected_keys or _raise(ValueError("unexpected keys"))
        acc['criterion'] = criterion
        acc['thresh'] = thr
        acc['by_image'] = bool(by_image)
        if bool(by_image):
            for k in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
                acc[k] /= n_images
        else:
            tp, fp, fn, n_true = acc['tp'], acc['fp'], acc['fn'], acc['n_true']
            sum_matched_score = acc['mean_true_score']

            mean_matched_score = _safe_divide(sum_matched_score, tp)
            mean_true_score    = _safe_divide(sum_matched_score, n_true)
            panoptic_quality   = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

            acc.update(
                precision          = precision(tp,fp,fn),
                recall             = recall(tp,fp,fn),
                accuracy           = accuracy(tp,fp,fn),
                f1                 = f1(tp,fp,fn),
                mean_true_score    = mean_true_score,
                mean_matched_score = mean_matched_score,
                panoptic_quality   = panoptic_quality,
            )

    accumulate = tuple(namedtuple('DatasetMatching',acc.keys())(*acc.values()) for acc in accumulate)
    return accumulate[0] if single_thresh else accumulate



# copied from scikit-image master for now (remove when part of a release)
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    max_label = int(label_field.max()) # Ensure max_label is an integer
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(max_label)
        label_field = label_field.astype(new_type)
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    new_max_label = offset - 1 + len(labels0)
    new_labels0 = np.arange(offset, new_max_label + 1)
    output_type = label_field.dtype
    required_type = np.min_scalar_type(new_max_label)
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        output_type = required_type
    forward_map = np.zeros(max_label + 1, dtype=output_type)
    forward_map[labels0] = new_labels0
    inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
    inverse_map[offset:] = labels0
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map



def group_matching_labels(ys, thresh=1e-10, criterion='iou'):
    """
    Group matching objects (i.e. assign the same label id) in a
    list of label images (e.g. consecutive frames of a time-lapse).

    Uses function `matching` (with provided `criterion` and `thresh`) to
    iteratively/greedily match and group objects/labels in consecutive images of `ys`.
    To that end, matching objects are grouped together by assigning the same label id,
    whereas unmatched objects are assigned a new label id.
    At the end of this process, each label group will have been assigned a unique id.

    Note that the label images `ys` will not be modified. Instead, they will initially
    be duplicated and converted to data type `np.int32` before objects are grouped and the result
    is returned. (Note that `np.int32` limits the number of label groups to at most 2147483647.)

    Example
    -------
    import numpy as np
    from stardist.data import test_image_nuclei_2d
    from stardist.matching import group_matching_labels

    _y = test_image_nuclei_2d(return_mask=True)[1]
    labels = np.stack([_y, 2*np.roll(_y,10)], axis=0)

    labels_new = group_matching_labels(labels)

    Parameters
    ----------
    ys : np.ndarray or list/tuple of np.ndarray
        list/array of integer labels (2D or 3D)
    
    """
    # check 'ys' without making a copy
    len(ys) > 1 or _raise(ValueError("'ys' must have 2 or more entries"))
    if isinstance(ys, np.ndarray):
        _check_label_array(ys, 'ys')
        ys.ndim > 1 or _raise(ValueError("'ys' must be at least 2-dimensional"))
        ys_grouped = np.empty_like(ys, dtype=np.int32)
    else:
        all(_check_label_array(y, 'ys') for y in ys) or _raise(ValueError("'ys' must be a list of label images"))
        all(y.shape==ys[0].shape for y in ys) or _raise(ValueError("all label images must have the same shape"))
        ys_grouped = np.empty((len(ys),)+ys[0].shape, dtype=np.int32)

    def _match_single(y_prev, y, next_id):
        y = y.astype(np.int32, copy=False)
        res = matching(y_prev, y, report_matches=True, thresh=thresh, criterion=criterion)
        # relabel dict (for matching labels) that maps label ids from y -> y_prev 
        relabel = dict(reversed(res.matched_pairs[i]) for i in res.matched_tps)
        y_grouped = np.zeros_like(y)
        for r in regionprops(y):
            m = (y[r.slice] == r.label)
            if r.label in relabel:
                y_grouped[r.slice][m] = relabel[r.label]
            else:
                y_grouped[r.slice][m] = next_id
                next_id += 1
        return y_grouped, next_id

    ys_grouped[0] = ys[0]
    next_id = ys_grouped[0].max() + 1
    for i in range(len(ys)-1):
        ys_grouped[i+1], next_id = _match_single(ys_grouped[i], ys[i+1], next_id)
    return ys_grouped



def _shuffle_labels(y):
    _check_label_array(y, 'y')
    y2 = np.zeros_like(y)
    ids = tuple(set(np.unique(y)) - {0})
    relabel = dict(zip(ids,np.random.permutation(ids)))
    for r in regionprops(y):
        m = (y[r.slice] == r.label)
        y2[r.slice][m] = relabel[r.label]
    return y2


def batch_matching(gt_files, seg_files, thresh_list=[0.5,0.625,0.75,0.875,1], downsample_factor=1,to_instance=False):
    """
    Calculate the matching metrics between ground truth and prediction images in a batch.

    Args:
        gt_files (list): A list of file paths to the ground truth images.
        seg_files (list): A list of file paths to the prediction images.
        thresh_list (list, optional): A list of threshold values for matching. Defaults to [0.5, 0.625, 0.75, 0.875, 1].
        downsample_factor (int, optional): The downsampling factor for the input images. Defaults to 1.
        to_instance (bool, optional): Whether to convert the ground truth images to instance labels. Defaults to False.

    Raises:
        ValueError: If the number of ground truth and prediction images do not match.

    Returns:
        pandas.DataFrame: A DataFrame containing the matching metrics for each image and threshold.
            The DataFrame has the following columns:
            - img_id (str): The ID of the image.
            - match_3d (float): The matching metric for the 3D images.

    """
    import tifffile
    import pandas as pd
    if len(gt_files) != len(seg_files):
        raise ValueError('number of ground truth and prediction images do not match')
    output_df = pd.DataFrame()
    for i in range(len(gt_files)):
        for idxt,threshold in enumerate(thresh_list):
            gt_img = tifffile.imread(gt_files[i])
            if to_instance:
                from skimage import measure
                gt_img = measure.label(gt_img,background=0)
                

            # results 3d
            seg_img = tifffile.imread(seg_files[i])
            match_3d = matching(gt_img,seg_img,  thresh=threshold)
            
            df_temp = pd.DataFrame([match_3d])
            df_temp['img_id'] = str(gt_files[i].split('/')[-1])
            output_df = pd.concat([output_df,df_temp])
    return output_df