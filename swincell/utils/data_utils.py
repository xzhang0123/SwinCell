# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os,glob, random

import libtiff
libtiff.libtiff_ctypes.suppress_warnings()
import numpy as np
import pandas as pd
import torch
from natsort import natsorted

from monai import data, transforms

from monai.transforms import  Transform, MapTransform


# from monai.utils.enums import TransformBackends

class flow_reshape(Transform):

    # backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img):
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # result =np.array(np.split(img,4,axis=2),dtype=np.uint8)
        result =np.array(np.split(img,4,axis=2))
        #test
        # result = 
        result[0] = np.uint8(result[0]>0)
        result[1:] = (result[1:] - 127)/127
        # print('output flow shape',result.shape, result.dtype)
        # result.shape,(4, 96, 512, 512)

        
        return result

class flow_reshaped(MapTransform):




    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = flow_reshape()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
        
def split_dataset(root_data_dir, split_ratios, shuffle=False, seed=0,dict_keys=True):
    
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
            test_datalist = [{'image':image,'label':label} for image,label in zip(test_images,test_labels)]

            return train_datalist, validation_datalist, test_datalist
    else:

        if len(split_ratios) == 2:
            return train_images, validation_images
        elif len(split_ratios) == 3:
            return train_images, validation_images, test_images
    

    


class Sampler(torch.utils.data.Sampler):
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

    print('folder loader for tiff images')
    import os
    import glob
    if args.fold is None:  # if not specified, use 80% for training, 20% for validation
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

    else:  # if specified, use 60% for training, 20% for validation, 20% for testing
        N =len(os.listdir(os.path.join(args.data_dir,'images')))
        print('length of datasets',N)
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
    print(len(train_img_full_paths),len(train_label_full_paths),len(valid_img_full_paths),len(valid_label_full_paths),'a')

    if args.dataset =='colon':

        img_spatial_size= (1300,1030,129)

    elif args.dataset =='allen':

        img_spatial_size= (900,600,64)

    elif args.dataset =='nanolive':

        img_spatial_size = (512,512,96)

    else:
        raise Warning("dataset not defined")
        transform_resize = None

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
            # transform_resize,
	        transforms.Resized(keys=["image", "label"],spatial_size=img_spatial_size),

            
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

            transforms.Resized(keys=["image", "label"],spatial_size=img_spatial_size),  #for nanolive
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
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"],threshold=1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        print('test mode NA')

    else:

        # datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        #if args.use_normal_dataset:
        if 1:
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
#modified from here
    
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
            # transforms.RandSpatialCropSamplesd(
            #         keys=["image","label"],
            #         roi_size=[args.roi_x, args.roi_y, args.roi_z],
            #         num_samples=2,
            #         random_center=True,
            #         random_size=False,
            #     ),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"],threshold=1),
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"),
            # transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        print('test mode NA')
        # test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        # test_ds = data.Dataset(data=test_files, transform=test_transform)
        # test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        # test_loader = data.DataLoader(
        #     test_ds,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=args.workers,
        #     sampler=test_sampler,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        # loader = test_loader
    else:

        # datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        #if args.use_normal_dataset:
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


# def datafold_read(datalist, basedir, fold=0, key="training"):

#     with open(datalist) as f:
#         json_data = json.load(f)

#     json_data = json_data[key]

#     for d in json_data:
#         for k, v in d.items():
#             if isinstance(d[k], list):
#                 d[k] = [os.path.join(basedir, iv) for iv in d[k]]
#             elif isinstance(d[k], str):
#                 d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

#     tr = []
#     val = []
#     for d in json_data:
#         if "fold" in d and d["fold"] == fold:
#             val.append(d)
#         else:
#             tr.append(d)

#     return tr, val


# def get_loader(args):
#     data_dir = args.data_dir
#     datalist_json = args.json_list
#     train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
#     train_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.CropForegroundd(
#                 keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
#             ),
#             transforms.RandSpatialCropd(
#                 keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
#             ),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#             transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )
#     val_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )

#     test_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )

#     if args.test_mode:

#         val_ds = data.Dataset(data=validation_files, transform=test_transform)
#         val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
#         test_loader = data.DataLoader(
#             val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
#         )

#         loader = test_loader
#     else:
#         train_ds = data.Dataset(data=train_files, transform=train_transform)

#         train_sampler = Sampler(train_ds) if args.distributed else None
#         train_loader = data.DataLoader(
#             train_ds,
#             batch_size=args.batch_size,
#             shuffle=(train_sampler is None),
#             num_workers=args.workers,
#             sampler=train_sampler,
#             pin_memory=True,
#         )
#         val_ds = data.Dataset(data=validation_files, transform=val_transform)
#         val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
#         val_loader = data.DataLoader(
#             val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
#         )
#         loader = [train_loader, val_loader]

#     return loader
