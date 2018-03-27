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
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
           'pant_length_labels']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FashionAIDataset(Dataset):

    def __init__(self, csv_file, img_dir, class_name="all", transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_file)
        df.columns = ['image_id', 'class', 'label']

        if class_name != "all":
            self.df_load = (df[df['class'] == class_name].copy())
            self.df_load.reset_index(inplace=True)
            del self.df_load['index']
        else:
            self.df_load = df

        n = len(self.df_load)
        self.n_class = len(self.df_load['label'][0])
        self.Y = np.zeros((n), dtype=int)
        for i in range(n):
            label = self.df_load['label'][i].find('y')
            self.Y[i] = label

    def __len__(self):
        return len(self.df_load)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df_load['image_id'][idx])
        img = pil_loader(img_name)

        # import ipdb; ipdb.set_trace()
        if self.transform:
            img = self.transform(img)

        return img, self.Y[idx]

class FashionAITestDataset(Dataset):

    def __init__(self, csv_file, img_dir, class_name="all", transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_file)
        df.columns = ['image_id', 'class', 'label']
        del df['label']

        if class_name != "all":
            self.df_load = (df[df['class'] == class_name].copy())
            self.df_load.reset_index(inplace=True)
            del self.df_load['index']
        else:
            self.df_load = df

    def __len__(self):
        return len(self.df_load)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df_load['image_id'][idx])
        img = pil_loader(img_name)

        if self.transform:
            img = self.transform(img)

        return img
