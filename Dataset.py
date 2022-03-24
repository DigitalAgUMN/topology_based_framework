#!/usr/bin/env python
# coding: utf-8
"""
@author: Chenxi
"""
import torch.utils as utils
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utility_functions import get_hist2d, get_target, get_separability
from utility_functions import get_coordinate

def scale_percentile_n(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)
    mins = np.percentile(matrix, 0, axis=0)
    maxs = np.percentile(matrix, 100, axis=0)
    matrix = (matrix - mins[None:]) / (maxs[None:] * 0.5) - 1
    matrix = np.reshape(matrix, [w, h, d]).astype(np.float32)
    matrix = matrix.clip(-1, 1)
    return matrix

class BuildingDataset(utils.data.Dataset):
    def __init__(self, dir, transform=None, scale=True, target=True):
        self.dir = dir
        self.transform = transform
        self.img_list = os.listdir(dir)
        self.scale = scale
        self.target = target
        if self.target:
            self.target_list = os.listdir(dir.replace('img', 'target'))

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        if self.img_list[index].endswith('tif'):
            image = io.imread(os.path.join(self.dir, self.img_list[index])).astype(np.int16)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]

            if self.scale:
                image = scale_percentile_n(image)
            sample = {}
            sample['image'] = image.transpose(2,0,1)
            sample['patch'] = self.img_list[index].split('.')[0]
            sample['name'] = self.img_list[index]
            if self.target:
                target = io.imread(os.path.join(self.dir.replace('img', 'target'), self.target_list[index])).transpose(
                    2, 0, 1)
                sample['label'] = target

            if self.transform:
                sample = self.transform(sample)
            return sample