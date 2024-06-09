# Copyright 2020 - 2021 MONAI Consortium
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
import os
import copy
from glob import glob
import json

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from .samplers import Sampler
from sklearn.model_selection import train_test_split

def get_loader_hecktor(args):
    data_dir = args.data_dir #e.g. /home/numansaeed/Projects/PCT-Net/dataset/HECKTOR/1_1_1_s176v2/


    datalist_json = os.path.join(args.json_list)

    # Load the data from the JSON file
    with open(datalist_json, 'r') as f:
        data_files = json.load(f)
    # Access the train and test sets
    train_files = data_files['train']
    test_files = data_files['test']

    # Add intial path from the data_dir to each value in the train_files and test_files
    # train file consists of key "train" with a list of dictionaries, each dictionary has keys "image" and "label"
    # test file consists of key "test" with a list of dictionaries, each dictionary has keys "image" and "label"
    train_files = [{"image": os.path.join(data_dir, i["image"]), "label": os.path.join(data_dir, i["label"])} for i in train_files]
    test_files = [{"image": os.path.join(data_dir, i["image"]), "label": os.path.join(data_dir, i["label"])} for i in test_files]

    # transforms for training
    train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        transforms.SpatialPadd(keys=["image",  "label"], spatial_size=(176,176,176), method='end'),
        transforms.Orientationd(keys=["image", "label"], axcodes="PLS"),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        transforms.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.20,
        ),
        transforms.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.20,
        ),
        transforms.RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.20,
        ),
        transforms.RandRotate90d(
            keys=["image", "label"],
            prob=0.20,
            max_k=3,
        ),
        transforms.ToTensord(keys=["image", "label"]),
    ]
)

    # transforms for validation/test mode
    val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=(176,176,176), method='end'),
        transforms.Orientationd(keys=["image", "label"], axcodes="PLS"),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True),
        transforms.ToTensord(keys=["image", "label"]),
    ]
)

    if args.gen_train_adv_mode:
        print('Loader: Mode = Generate Train-Adv Mode:  Training Data is Loaded')

        dataset = data.Dataset(data=train_files, transform=val_transform)

        sampler = Sampler(dataset, shuffle=False) if args.distributed else None

        loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True,
        )


    elif args.test_mode or args.gen_val_adv_mode:
        print('\nLoader: Mode = Test Mode or Generate Val-Adv Mode:  Validation Data is Loaded')
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            # num_workers=args.workers,
            sampler=test_sampler,
            # pin_memory=True,
            # persistent_workers=True,
        )
        loader = test_loader


    else:
        print('\nLoader: Both Train and Test Data are Loaded')

        if args.use_normal_dataset:
            train_ds = data.Dataset(data=train_files, transform=train_transform)
        else:
            train_ds = data.CacheDataset(data=train_files, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers)

        train_sampler = Sampler(train_ds) if args.distributed else None

        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )

        val_ds = data.Dataset(data=test_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None

        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )

        loader = [train_loader, val_loader]

    return loader