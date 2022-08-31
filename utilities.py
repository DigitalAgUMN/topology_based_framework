#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
from skimage import io
import os
from torchvision.utils import make_grid

np.random.seed(1003)
def get_hist2d(img,
               mask=None,
               label=None,
               bins=129,
               scales=[0.3, 0.6],
               b_invert_yaxis=True):
    '''
    Generate the 2D heat map
    Args:
        img (ndarray), 4D numpy array (batch-channel-height-width), the input image that you want to generate heat maps for
        mask (ndarray), optional, the mask for the input image
        label (ndarray), optional, pixel-wise label information of the input image
        bins (int), bin size for the heat map
        scales (list), scales for the heat map
    '''
    if label is None:
        label = np.zeros((img.shape[2], img.shape[3]))
    code_class = np.unique(label)
    n_class = len(code_class)
    n_img = img.shape[0]
    list_img = []
    list_xedges, list_yedges = [], []
    for i_img in range(n_img):
        arr_cur = img[i_img].transpose((1, 2, 0))
        list_class = []
        list_xedges_class, list_yedges_class = [], []
        for i_class in range(0, n_class):
            ## return point pair (n by 2)
            data_flat = arr_cur[label == i_class]
            if mask is not None:
                mask = mask[label == i_class]
                data_flat = data_flat[mask == 1]
            indi_valid = np.logical_and(data_flat[:, 0] != 0, data_flat[:, 1] != 0)
            data_flat = data_flat[indi_valid, :]
            if not isinstance(bins, list):
                bins0 = np.linspace(0, scales[0], bins)
                bins1 = np.linspace(0, scales[1], bins)
                new_bins = [bins1, bins0]
            if isinstance(bins, list):
                new_bins = bins
            data_hist, yedges, xedges = np.histogram2d(
                data_flat[:, 1], data_flat[:, 0], bins=new_bins
            )
            if b_invert_yaxis:
                data_hist = data_hist[::-1]
                yedges = yedges[::-1]
            list_class.append(data_hist)
            list_xedges_class.append(xedges)
            list_yedges_class.append(yedges)
        list_img.append(list_class)
        list_xedges.append(list_xedges_class)
        list_yedges.append(list_yedges_class)
    return np.array(list_img), np.array(list_xedges), np.array(list_yedges)

def JM_distance(x, y):
    '''
    Calculate the Jeffries-Matusita Distance between x and y
    x and y have the same number of variables (columns).
    Each row is an observation.
    '''
    dif_mean = np.empty((1, x.shape[1]))
    for i in range(x.shape[1]):
        dif_mean[0, i] = x[:, i].mean() - y[:, i].mean()
    comatrix_x = np.cov(x, rowvar=False)
    comatrix_y = np.cov(y, rowvar=False)
    comatrix_mean = 0.5 * (comatrix_x + comatrix_y)
    alpha = (
            0.125 * np.dot(
        np.dot(dif_mean, np.linalg.inv(comatrix_mean)),
        dif_mean.T
    )
            + 0.5 * np.log(
        np.linalg.det(comatrix_mean) /
        np.sqrt(np.linalg.det(comatrix_x) * np.linalg.det(comatrix_y))
    )
    )
    output = np.sqrt(2 * (1 - np.exp(-alpha)))[0, 0]
    return (output)

def get_separability(arr, arr_class):
    """
    :param arr:
    :param arr_class:
    :return:
    """
    code_class = np.unique(arr_class)
    n_class = len(code_class)
    n_img = arr.shape[0]
    list_separability = []
    for i_img in range(n_img):
        arr_cur = arr[i_img].transpose((1, 2, 0))
        list_class = []
        for i_class in range(n_class):
            for j_class in range(n_class):
                indi_valid = ~(np.isnan(arr_cur[:, :, 0]) | np.isnan(arr_cur[:, :, 1]))
                data_flat_pos = arr_cur[(arr_class == code_class[i_class]) & indi_valid]
                data_flat_neg = arr_cur[(arr_class == code_class[j_class]) & indi_valid]
                separability = JM_distance(data_flat_pos, data_flat_neg)
                list_class.append(separability)
        list_separability.append(list_class)
    return np.delete(np.unique(np.array(list_separability)), [0])

def get_target(
        data_hist,
        percentile_pure=50,
        crop='corn',
        separability=None,
        threshold_separability=None):
    n_img, n_class, n_y, n_x = data_hist.shape
    list_img = []
    for i_img in range(n_img):
        '''
        1. find which class has most pixels in each grid
        2. data_hist for image at each date should have size of (classes, bins1, bins2) in our case, class is 3
        3. for each pixel in the bins1-bins2 grid, three classes 
        could all have values, which means there's overlap in the feature combination
        4. when all classes have no value in a grid, the idx_max is assigned to 0
        '''
        idx_max = np.argmax(data_hist[i_img], axis=0)
        list_class = []
        crop_index = {
            'background': [0],
            'corn': [1],
            'soybean': [2],
            'all': [0,1,2]
        }
        i_class_list = crop_index[crop]
        for i_class in i_class_list:
            if separability.mean() > threshold_separability:
                indi_cur = (idx_max == i_class)
                ## non-zero pixel in 2d histogram
                indi_pos = 0 < data_hist[i_img, i_class]
                ## make sure the target is the class that has the largest value and exclude grids having no values
                data_target_cur = data_hist[i_img, i_class] * indi_cur * indi_pos
                data_flat_cur = data_hist[i_img, i_class][indi_cur & indi_pos]
                data_flat_cur = np.sort(data_flat_cur)
                cumsum_flat_cur = np.cumsum(data_flat_cur)
                try:
                    cumsum_pure = cumsum_flat_cur[-1] * (100 - percentile_pure) / 100
                    idx_threshold_pure = int(np.clip(
                        np.where(cumsum_flat_cur > cumsum_pure)[0][0],
                        0, len(data_flat_cur) - 1
                    ))
                    threshold_pure = data_flat_cur[idx_threshold_pure]
                    candidate = data_target_cur > threshold_pure
                    data_target_cur[candidate] = i_class
                    data_target_cur[~candidate] = 0
                except:
                    pass
            else:
                data_target_cur = np.zeros((n_y, n_x))
            list_class.append(data_target_cur)
        list_img.append(list_class)
    return np.array(list_img)

def get_coordinate(arr, x_coor=True):
    coor1 = arr.copy().squeeze()
    coor1[coor1 != 0] = 1
    coor2 = np.zeros_like(arr).squeeze()
    row, col = arr.shape[2], arr.shape[3]
    if x_coor:
        for i in range(row):
            try:
                coor2[i, :] = np.arange(0, col)
            except:
                pass
    if not x_coor:
        for i in range(col):
            coor2[:, i] = np.arange(0, row)
    return coor1 * coor2







