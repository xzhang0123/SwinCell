import os
import glob
import random
import math
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from monai import data, transforms
from monai.transforms import  Transform, MapTransform
from swincell.cellpose_dynamics import masks_to_flows
from skimage import measure
# import libtiff
# libtiff.libtiff_ctypes.suppress_warnings()
import warnings
warnings.filterwarnings('ignore')





# from monai.utils.enums import TransformBackends

class flow_reshape(Transform):
    """
    Transform to reshape the generated flow images for cell segmentation in PyTorch.

    This transform reshapes and processes 3D cellular segmentation images by splitting the input image
    along the third axis into four parts, where the first part is binarized and the remaining parts 
    are normalized. 

    Args:
        img (ndarray): Input image with dimensions (C, Z, X, Y). If the image has a single channel dimension,
                       it will be squeezed.

    Returns:
        ndarray: Transformed image with shape (4, Z, X, Y), where the first slice is a binary mask and 
                 the remaining slices are normalized flows.
    """
    def __call__(self, img):
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # result =np.array(np.split(img,4,axis=2),dtype=np.uint8)
        result =np.array(np.split(img,4,axis=2))

        result[0] = np.uint8(result[0]>0) 
        result[1:] = (result[1:] - 127)/127 # normalize flows to 0
        # print('output flow shape',result.shape, result.dtype)
        # result.shape,(czxy)

        
        return result

class flow_reshaped(MapTransform):

    """
    Transform to reshape the generated flow images for cell segmentation in PyTorch.

    This transform reshapes and processes 3D cellular segmentation images by splitting the input image
    along the third axis into four parts, where the first part is binarized and the remaining parts 
    are normalized. 

    Args:
        img (ndarray): Input image with dimensions (C, Z, X, Y). If the image has a single channel dimension,
                       it will be squeezed.

    Returns:
        ndarray: Transformed image with shape (4, Z, X, Y), where the first slice is a binary mask and 
                 the remaining slices are normalized flows.
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = flow_reshape()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    
class flow_generation(Transform):
    
    """
    Generate flow from image label for model training


    Args:
        img (ndarray): Input image with dimensions (C, Z, X, Y). If the image has a single channel dimension,
                       it will be squeezed.

    Returns:
        ndarray: Transformed image with shape (4, Z, X, Y), where the first slice is a binary mask and 
                 the remaining slices are normalized flows.
    """
    def __call__(self, labels):
        # if img has channel dim, squeeze it
        if labels.ndim == 4 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        labels = measure.label(labels)
        if np.max(labels) == 0:
            print('got empty masks, returning zeros')

            return np.zeros((4, *labels.shape),dtype=np.float32)
        label_with_flows = masks_to_flows(labels)
        results = np.concatenate((labels[None, :], label_with_flows),axis=0)
        results[0] = np.uint8(results[0]>0) 
        return np.float32(results)
    
class flow_generationd(MapTransform):

    """
    Generate flow from image label for model training
    Args:
        img (ndarray): Input image with dimensions (C, Z, X, Y). If the image has a single channel dimension,
                       it will be squeezed.

    Returns:
        ndarray: Transformed image with shape (4, Z, X, Y), where the first slice is a binary mask and 
                 the remaining slices are normalized flows.
    """
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = flow_generation()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
        
def split_dataset(root_data_dir, split_ratios, shuffle=False, seed=0,dict_keys=True):
    """
    Splits the dataset into training, validation, and (optional) test sets based on the given split ratios.
    
    Args:
        root_data_dir (str): The root directory of the dataset.
        split_ratios (List[float]): The ratios of the dataset to be split into training, validation, and (optional) test sets.
        shuffle (bool, optional): Whether to shuffle the files before splitting. Defaults to False.
        seed (int, optional): The seed value for shuffling the files. Defaults to 0.
        dict_keys (bool, optional): Whether to return the split datasets as lists of dictionaries with 'image' and 'label' keys. Defaults to True.
    
    Returns:
        Union[Tuple[List[Dict[str, str]], List[Dict[str, str]]], Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]]: If `dict_keys` is True, returns a tuple containing the training and validation datasets as lists of dictionaries with 'image' and 'label' keys. If `dict_keys` is False, returns a tuple containing the training and validation datasets as lists of file paths. If `split_ratios` contains three values, also returns a test dataset as a list of file paths.
    
    Raises:
        ValueError: If the sum of `split_ratios` is not equal to 1.
        ValueError: If the number of images and labels does not match.
    
    """
    
    assert sum(split_ratios) == 1, "Split ratios must sum to 1"

    
    all_images = natsorted(glob.glob(os.path.join(root_data_dir,'images/*tif*')))
    all_labels = natsorted(glob.glob(os.path.join(root_data_dir,'masks_with_flows/*tif*')))
    assert len(all_images) == len(all_labels), "Number of images and labels must match"

    if shuffle:       #shuffle the files to ensure random distribution
        random.seed(seed)
        random.shuffle(all_images)
        random.shuffle(all_labels)

    
    # Determine the split index
    if len(split_ratios) == 2:
        split_index = int(len(all_images) * split_ratios[0])
        train_images = all_images[:split_index]
        validation_images = all_images[split_index:]
        train_labels = all_labels[:split_index]
        validation_labels = all_labels[split_index:]
    elif len(split_ratios) == 3:
        train_end_index= int(len(all_images) * split_ratios[0])
        val_end_index = int(len(all_images) * split_ratios[1])

        train_images = all_images[:train_end_index]
        validation_images = all_images[train_end_index:val_end_index]
        test_images = all_images[val_end_index:]

        train_labels = all_labels[:train_end_index]
        validation_labels = all_labels[train_end_index:val_end_index]
        test_labels = all_labels[val_end_index:]

    else:
        raise ValueError("Invalid split ratios")
    
    if dict_keys:

        if len(split_ratios) == 2:
            train_datalist = [{'image':image,'label':label} for image,label in zip(train_images,train_labels)] 
            validation_datalist = [{'image':image,'label':label} for image,label in zip(validation_images,validation_labels)]

            return train_datalist, validation_datalist
        elif len(split_ratios) == 3:
            train_datalist= [{'image':image,'label':label} for image,label in zip(train_images,train_labels)]
            validation_datalist = [{'image':image,'label':label} for image,label in zip(validation_images,validation_labels)]
            test_datalist = [{'image':image} for image in test_images]

            return train_datalist, validation_datalist, test_datalist
    else:

        if len(split_ratios) == 2:
            return train_images, validation_images
        elif len(split_ratios) == 3:
            return train_images, validation_images, test_images
    
def split_dataset_folder(all_images, all_labels, split_ratios, shuffle=False, seed=0,dict_keys=True):
    """
    Splits the dataset into training, validation, and testing sets based on the given split ratios.
    
    Args:
        all_images (list): A list of all the image file paths.
        all_labels (list): A list of all the label file paths.
        split_ratios (list): A list of floats representing the ratios for splitting the dataset.
        shuffle (bool, optional): Whether to shuffle the files before splitting. Defaults to False.
        seed (int, optional): The seed value for shuffling. Defaults to 0.
        dict_keys (bool, optional): Whether to return the data as a list of dictionaries with 'image' and 'label' keys. Defaults to True.
    
    Raises:
        AssertionError: If the sum of split ratios is not equal to 1 or if the number of images and labels do not match.
    
    Returns:
        If dict_keys is True:
            - If split_ratios contains 2 elements:
                - train_datalist (list): A list of dictionaries containing 'image' and 'label' keys for the training set.
                - validation_datalist (list): A list of dictionaries containing 'image' and 'label' keys for the validation set.
            - If split_ratios contains 3 elements:
                - train_datalist (list): A list of dictionaries containing 'image' and 'label' keys for the training set.
                - validation_datalist (list): A list of dictionaries containing 'image' and 'label' keys for the validation set.
                - test_datalist (list): A list of dictionaries containing 'image' keys for the testing set.
        
        If dict_keys is False:
            - If split_ratios contains 2 elements:
                - train_images (list): A list of image file paths for the training set.
                - validation_images (list): A list of image file paths for the validation set.
            - If split_ratios contains 3 elements:
                - train_images (list): A list of image file paths for the training set.
                - validation_images (list): A list of image file paths for the validation set.
                - test_images (list): A list of image file paths for the testing set.
    """
    
    assert sum(split_ratios) == 1, "Split ratios must sum to 1"
    assert len(all_images) == len(all_labels), "Number of images and labels must match"

    if shuffle:       #shuffle the files to ensure random distribution
        random.seed(seed)
        random.shuffle(all_images)
        random.shuffle(all_labels)

    
    # Determine the split index
    if len(split_ratios) == 2:
        split_index = int(len(all_images) * split_ratios[0])
        train_images = all_images[:split_index]
        validation_images = all_images[split_index:]
        train_labels = all_labels[:split_index]
        validation_labels = all_labels[split_index:]
    elif len(split_ratios) == 3:
        train_end_index= int(len(all_images) * split_ratios[0])
        val_end_index = int(len(all_images) * split_ratios[1])

        train_images = all_images[:train_end_index]
        validation_images = all_images[train_end_index:val_end_index]
        test_images = all_images[val_end_index:]

        train_labels = all_labels[:train_end_index]
        validation_labels = all_labels[train_end_index:val_end_index]
        test_labels = all_labels[val_end_index:]

    else:
        raise ValueError("Invalid split ratios")
    
    if dict_keys:

        if len(split_ratios) == 2:
            train_datalist = [{'image':image,'label':label} for image,label in zip(train_images,train_labels)] 
            validation_datalist = [{'image':image,'label':label} for image,label in zip(validation_images,validation_labels)]

            return train_datalist, validation_datalist
        elif len(split_ratios) == 3:
            train_datalist= [{'image':image,'label':label} for image,label in zip(train_images,train_labels)]
            validation_datalist = [{'image':image,'label':label} for image,label in zip(validation_images,validation_labels)]
            test_datalist = [{'image':image} for image in test_images]

            return train_datalist, validation_datalist, test_datalist
    else:

        if len(split_ratios) == 2:
            return train_images, validation_images
        elif len(split_ratios) == 3:
            return train_images, validation_images, test_images
    


class Sampler(torch.utils.data.Sampler):
    # sampler from monai package, Copyright 2020 - 2022 MONAI Consortium
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch




def folder_loader(args):
    """
    Load the data for training, validation, or inference.

    Parameters:
        args (Namespace): The command line arguments containing the data directory, fold, inference flag, and other parameters.

    Returns:
        list: A list containing the train and validation data loaders if not in inference mode.
        DataLoader: The test data loader if in inference mode.

    Raises:
        Warning: If the dataset is not defined.

    Note:
        - If the `fold` argument is not specified, the data is split into training and validation sets with a ratio of 80% and 20% respectively.
        - If the `fold` argument is specified, the data is split into training, validation, and testing sets with ratios of 60%, 20%, and 20% respectively.
        - The data is loaded from the specified data directory.
        - The data is transformed using the specified transformations.
        - The data loaders are created with the specified batch size and number of workers.
    """

    print('folder loader for tiff images')
    import os
    import glob
    if not args.test_mode:  # if not specified, use 80% for training, 20% for validation
        if args.fold is None:
            N = len(os.listdir(os.path.join(args.data_dir,'images')))
            img_full_paths = natsorted(glob.glob(os.path.join(args.data_dir,'images/*.tif*')))
            label_full_paths = natsorted(glob.glob(os.path.join(args.data_dir,'masks_with_flows/*.tif*')))

            if len(img_full_paths)<5:
                img_full_paths = img_full_paths * 10
                label_full_paths = label_full_paths * 10
            # split dataset, 80% for training, 20% for validation   
            valid_img_full_paths = img_full_paths[::5]
            valid_label_full_paths = label_full_paths[::5]

            train_img_full_paths = [f for i,f in enumerate(img_full_paths) if i%5 != 0]
            train_label_full_paths = [f for i,f in enumerate(label_full_paths) if i%5 != 0]
        # end of split dataset
        else:
         # if specified, use 60% for training, 20% for validation, 20% for testing
            N =len(os.listdir(os.path.join(args.data_dir,'images')))
            # print('length of datasets',N)
            # whole dataset
            img_full_paths = natsorted(glob.glob(os.path.join(args.data_dir,'images/*.tif*')))
            label_full_paths = natsorted(glob.glob(os.path.join(args.data_dir,'masks_with_flows/*.tif*')))
            #------split -------
            valid_img_full_paths = [f for i,f in enumerate(img_full_paths) if i%5 == args.fold-1]
            valid_label_full_paths = [f for i,f in enumerate(label_full_paths) if i%5 == args.fold-1]

            test_img_full_paths = [f for i,f in enumerate(img_full_paths) if i%5 == args.fold]
            test_label_full_paths = [f for i,f in enumerate(label_full_paths) if i%5 == args.fold]

            train_img_full_paths = [f for f in img_full_paths if f not in valid_img_full_paths and f not in test_img_full_paths]
            train_label_full_paths = [f for f in label_full_paths if f not in valid_label_full_paths and f not in test_label_full_paths]
            print('length of train/valid/test datasets',len(train_img_full_paths),len(train_label_full_paths),len(valid_img_full_paths),len(valid_label_full_paths),len(test_img_full_paths),len(test_label_full_paths))

        train_datalist = [{'image':image,'label':label} for image,label in zip(train_img_full_paths, train_label_full_paths)]    
        val_datalist = [{'image':image,'label':label} for image,label in zip(valid_img_full_paths, valid_label_full_paths)]  
        print('length of train/valid',len(train_img_full_paths),len(train_label_full_paths),len(valid_img_full_paths),len(valid_label_full_paths))

        if args.dataset =='colon':

            img_shape= (1300,1030,129) #original shape
            img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
            img_reshape = tuple(int(e) for e in img_reshape)

        elif args.dataset =='allen':

            img_shape=(900,600,64)
            img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
            img_reshape = tuple(int(e) for e in img_reshape)

        elif args.dataset =='nanolive':

            img_shape=(512,512,96)
            img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
            img_reshape = tuple(int(e) for e in img_reshape)

        else:
            raise Warning("dataset not defined")
            img_reshape = None

        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                flow_reshaped(keys=["label"]),
                #----------------------------for multichannel image-----------------------
                # transforms.AddChanneld(keys=["image"]),
                # transforms.ConvertToMultiChannelNanolived(keys="label"),
                #----------------------------for single channel image-----------------------
                # transforms.AddChanneld(keys=["image","label"]),
                # transforms.AsDiscreted(keys=["label"],threshold=1),
                #----------------------------------------------------------------
                transforms.Resized(keys=["image", "label"],spatial_size=img_reshape),

                
                # transforms.RandZoomd(keys=["image", "label"],prob=0.5,min_zoom=0.85,max_zoom=1.15),
                # transforms.Spacingd(
                #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.RandSpatialCropSamplesd(
                    keys=["image","label"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),
                # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                flow_reshaped(keys=["label"]),
                #----------------------------for multichannel-----------------------
                # transforms.AddChanneld(keys=["image"]),
                # transforms.ConvertToMultiChannelNanolived(keys="label"),
                #----------------------------for single channel-----------------------
                # transforms.AddChanneld(keys=["image","label"]),
                # transforms.AsDiscreted(keys=["label"],threshold=1),
                #----------------------------------------------------------------

                transforms.Resized(keys=["image", "label"],spatial_size=img_reshape),  #for nanolive
                # transforms.RandZoomd(keys=["image", "label"],prob=0.5,min_zoom=0.85,max_zoom=1.05),
                # transforms.Spacingd(
                #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),   # for  anisotropic datasets
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.RandSpatialCropSamplesd(
                    keys=["image","label"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=2,
                    random_center=True,
                    random_size=False,
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        if 1: #no cache
            train_ds = data.Dataset(data=train_datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=train_datalist, transform=train_transform, cache_num=1, cache_rate=0.2, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_datalist, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

        return loader
    else:   # Inference mode
        print('Inference mode')
        img_full_paths = natsorted(glob.glob(os.path.join(args.data_folder,'images/*.tif*')))
        # label_full_paths = natsorted(glob.glob(os.path.join(args.data_folder,'masks_with_flows/*.tif*')))
        #------split -------
        test_img_full_paths = [f for i,f in enumerate(img_full_paths)]
        # test_label_full_paths = [f for i,f in enumerate(label_full_paths)]
        test_datalist =  [{'image':image} for image in test_img_full_paths]
        
        test_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0, b_max=1, clip=True
                ),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        test_ds = data.Dataset(data=test_datalist, transform=test_transform)
        test_loader = data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=None,
        )
        return test_loader





def folder_loader_nothefly(args):
    """
    Load data from a folder and return a list of data loaders for training and validation.
    
    Args:
        args (Namespace): 
    Returns:
        list: A list containing two data loaders: train_loader and val_loader.
            - train_loader (DataLoader): The data loader for training.
            - val_loader (DataLoader): The data loader for validation.
    """
    
    print('folder loader for tiff images')
    import os
    import glob

    image_files = natsorted(glob.glob(args.data_dir +'images/*tif*'))
    mask_files = natsorted(glob.glob(args.data_dir +'labels/*tif*'))#
    train_datalist, val_datalist = split_dataset_folder(image_files,mask_files, split_ratios=[0.8,0.2])
    if args.dataset =='colon':

        img_shape= (1300,1030,129) #original shape
        img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
        img_reshape = tuple(int(e) for e in img_reshape)

    elif args.dataset =='allen':

        img_shape=(900,600,64)
        img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
        img_reshape = tuple(int(e) for e in img_reshape)

    elif args.dataset =='nanolive':

        img_shape=(512,512,96)
        img_reshape = (img_shape[0]//args.downsample_factor,img_shape[1]//args.downsample_factor,img_shape[2]//args.downsample_factor)
        img_reshape = tuple(int(e) for e in img_reshape)

    else:
        raise Warning("dataset not defined")
        img_reshape = None

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            flow_reshaped(keys=["label"]),
            #----------------------------for multichannel image-----------------------
            # transforms.AddChanneld(keys=["image"]),
            # transforms.ConvertToMultiChannelNanolived(keys="label"),
            #----------------------------for single channel image-----------------------
            # transforms.AddChanneld(keys=["image","label"]),
            # transforms.AsDiscreted(keys=["label"],threshold=1),
            #----------------------------------------------------------------
	        transforms.Resized(keys=["image", "label"],spatial_size=img_reshape),

            
            # transforms.RandZoomd(keys=["image", "label"],prob=0.5,min_zoom=0.85,max_zoom=1.15),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.RandSpatialCropSamplesd(
                keys=["image","label"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            flow_reshaped(keys=["label"]),
            #----------------------------for multichannel-----------------------
            # transforms.AddChanneld(keys=["image"]),
            # transforms.ConvertToMultiChannelNanolived(keys="label"),
            #----------------------------for single channel-----------------------
            # transforms.AddChanneld(keys=["image","label"]),
            # transforms.AsDiscreted(keys=["label"],threshold=1),
            #----------------------------------------------------------------

            transforms.Resized(keys=["image", "label"],spatial_size=img_reshape),  #for nanolive
            # transforms.RandZoomd(keys=["image", "label"],prob=0.5,min_zoom=0.85,max_zoom=1.05),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),   # for  anisotropic datasets
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.RandSpatialCropSamplesd(
                keys=["image","label"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0, b_max=1, clip=True
            ),
            transforms.ToTensord(keys=["image"]),
        ]
    )


    if 1: #no cache
        train_ds = data.Dataset(data=train_datalist, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=train_datalist, transform=train_transform, cache_num=1, cache_rate=0.2, num_workers=args.workers
        )
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_ds = data.Dataset(data=val_datalist, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )
    loader = [train_loader, val_loader]

    return loader

def get_loader_Allen_tiff(args):
    """
    Get the data loaders for the Allen tiff dataset.

    Args:
        args (Namespace): The command-line arguments.

    Returns:
        list: A list containing the train and validation data loaders.

    This function loads the Allen tiff dataset by reading the metadata from the csv file. It then splits the dataset into training and validation sets based on the specified fold. The training and validation sets are created by joining the image and label paths. The data is then transformed using a series of transformations defined in the `train_transform`, `val_transform`, and `test_transform` variables. The transformed data is then loaded into data loaders using the `data.Dataset` and `data.DataLoader` classes. The data loaders are returned as a list.
    """
    
    root_dir = '/data/download_data/quilt-data-access-tutorials-main/all_fov/'
    df = pd.read_csv(root_dir+'meta_info.csv')

    fold = args.fold
    N =50
    print('getloader allen tiff')
    Nd = int(N/5) #5 fold training
    input_dir=df['fov_path'].unique()[:N].tolist()
    target_dir=df['fov_seg_path'].unique()[:N].tolist()
    print(fold,Nd)
    valid_img_paths =input_dir[fold*Nd:(fold+1)*Nd]
    valid_label_paths = target_dir[fold*Nd:(fold+1)*Nd]
    train_img_paths =[path for path in input_dir if path not in valid_img_paths]
    train_label_paths =[path for path in target_dir if path not in valid_label_paths]

    

    train_img_full_paths =[os.path.join(root_dir,'fov_path_channel/'+file.split('/')[-1]) for file in train_img_paths]
    train_label_full_paths =[os.path.join(root_dir,'fov_seg_path_channel/'+file.split('/')[-1]) for file in train_label_paths]
    valid_img_full_paths =[os.path.join(root_dir,'fov_path_channel/'+file.split('/')[-1]) for file in valid_img_paths]
    valid_label_full_paths =[os.path.join(root_dir,'fov_seg_path_channel/'+file.split('/')[-1]) for file in valid_label_paths]
    #
    print(len(train_img_full_paths),len(train_label_full_paths),len(valid_img_full_paths),len(valid_label_full_paths),'a')

    # datalist0 =sorted(glob.glob(os.path.join(image_dir+'/*.tiff')))
    # datalist0 =[{'image':i} for i in datalist0[:30]] +
    # datadic ={'image':i for i in datalist}

    train_datalist = [{'image':a,'label':b} for a,b in zip(train_img_full_paths,train_label_full_paths)]    

    val_datalist = [{'image':a,'label':b} for a,b in zip(valid_img_full_paths,valid_label_full_paths)]  

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"],threshold=1),
            transforms.AddChanneld(keys=["image", "label"]),
	        transforms.Resized(keys=["image", "label"],spatial_size=(512,512,96)),
            transforms.RandZoomd(keys=["image", "label"],prob=0.5,min_zoom=0.85,max_zoom=1.15),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.RandSpatialCropSamplesd(
                keys=["image","label"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=2,
                random_center=True,
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"],threshold=1),
            transforms.AddChanneld(keys=["image", "label"]),
	        transforms.Resized(keys=["image", "label"],spatial_size=(512,512,96)),
            # transforms.RandZoomd(keys=["image", "label"],prob=0.5,min_zoom=0.85,max_zoom=1.05),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
       
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"],threshold=1),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if 0:
        train_ds = data.Dataset(data=train_datalist, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=train_datalist, transform=train_transform, cache_num=24, cache_rate=1, num_workers=args.workers
        )
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_ds = data.Dataset(data=val_datalist, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )
    loader = [train_loader, val_loader]

    return loader

