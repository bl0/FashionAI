#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 'pant_length_labels']
classes_len = {'collar_design_labels':[0,5],
               'neckline_design_labels':[5,15],
               'skirt_length_labels':[15,21],
               'sleeve_length_labels':[21,30],
               'neck_design_labels':[30,35],
               'coat_length_labels':[35,43],
               'lapel_design_labels':[43,48],
               'pant_length_labels':[48,54]}
all_class_lengths = 54

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FashionAIVisibleDataset(Dataset):

    def __init__(self, csv_file, img_dir, input_size, stage='train', class_name="all", transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.stage = stage
        self.input_size = input_size
        self.class_name = class_name

        df = pd.read_csv(csv_file, header=None)
        df.columns = ['image_id', 'class', 'label']

        if class_name != "all":
            self.df_load = (df[df['class'] == class_name].copy())
            self.df_load.reset_index(inplace=True)
            del self.df_load['index']
        else:
            self.df_load = df

        n = len(self.df_load)

        self.Y = np.zeros((n), dtype=int)
        self.idx_visible = []
        self.idx_invisible = []
        for i in range(n):
            if self.df_load['label'][i][0] == 'y':
                self.idx_invisible.append(i)
                self.Y[i] = 0
            else:
                self.idx_visible.append(i)
                self.Y[i] = 1

    def __len__(self):
        return 2*len(self.idx_visible) if self.stage == 'train' else len(self.df_load)

    def __getitem__(self, idx):
        if self.stage == 'train':
            if len(self.idx_invisible) == 0 or idx % 2 == 0:
                idx = self.idx_visible[int(idx / 2)]
            else:
                idx = self.idx_invisible[int(idx / 2) % len(self.idx_invisible)]
        img_name = os.path.join(self.img_dir, self.df_load['image_id'][idx])
        img = pil_loader(img_name)

        if self.Y[idx] or self.stage != 'train':
            img = transforms.Resize((self.input_size, self.input_size))(img)
        else:
            img = transforms.RandomResizedCrop(self.input_size)(img)

        if self.transform:
            img = self.transform(img)

        return img, self.Y[idx], classes.index(self.df_load['class'][idx])

class FashionAITestDataset(Dataset):

    def __init__(self, csv_file, img_dir, class_name="all", transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_file, header=None)
        df.columns = ['image_id', 'class', 'label']
        del df['label']

        self.class_name = class_name
        if class_name != "all":
            self.df_load = (df[df['class'] == class_name].copy())
            self.df_load.reset_index(inplace=True)
            del self.df_load['index']
        else:
            self.df_load = df

        n = len(self.df_load)
        self.idx = np.zeros((n, all_class_lengths), dtype=int)
        for i in range(n):
            left, right = classes_len[self.df_load['class'][i]]
            self.idx[i, left:right] = 1

    def __len__(self):
        return len(self.df_load)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df_load['image_id'][idx])
        img = pil_loader(img_name)

        if self.transform:
            img = self.transform(img)

        return img, self.idx[idx]
