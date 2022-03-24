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
                    ## if you want to show background as well, please assign the value as the number of class
                    ## ohterwise, use 0
                    data_target_cur[~candidate] = 0
                except:
                    pass
                #                 data_target_cur = data_target_cur / threshold_pure
            else:
                # data_target_cur = np.ones((n_y, n_x)) * 3
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


def confusion_matrix(pred, target, classes):
    assert isinstance(classes, list), 'please input a class list'
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().detach().numpy()
    class_num = len(classes)
    cm = np.zeros([class_num, class_num])
    pred = pred.squeeze()
    target = target.squeeze()
    for pred_c in classes:
        for targ_c in classes:
            cm[targ_c, pred_c] = ((pred == pred_c) * (target == targ_c)).sum()
    return cm


def project_to_target(arr,
                      patch_index,
                      patch_size,
                      id,
                      classes=[0, 1],
                      use_cm=False,
                      use_mask=True):
    '''
    :param arr: the predicted target, should be integer
    :param patch_index:
    :param patch_size:
    :return:
    '''

    b, h, w = arr.shape
    arr = arr.cpu().data.numpy()
    pred = np.zeros([b, patch_size, patch_size])
    h_interval = 0.6 / h
    w_interval = 0.3 / h
    cm = np.zeros([len(classes), len(classes)])
    for idx_b in range(0, b):
        '''
        get corresponding CDL
    '''
        date = patch_index[idx_b].split('_')[0]
        pt = patch_index[idx_b].replace(date + '_', '', 1)
        try:
            raw_img = io.imread(
                os.path.join(r'F:\DigitalAG\liheng\EU\2019\segmentation(wt_sfl_cn)\\'+ str(date), str(pt) + '.tif'))
            target = io.imread(
                os.path.join(r'F:\DigitalAG\liheng\EU\2019\segmentation(wt_sfl_cn)\target', str(pt) + '.tif'))
        except:
            raw_img = io.imread(
                os.path.join(r'F:\DigitalAG\liheng\EU\2018\segmentation(' + id + ')2\\'+ str(date), str(pt) + '.tif'))
            target = io.imread(
                os.path.join(r'F:\DigitalAG\liheng\EU\2018\segmentation(' + id + ')2'+ r'\target', str(pt) + '.tif'))

        arr_cur = arr[idx_b].squeeze()
        for c in classes[1:]:
            coord = np.argwhere(arr_cur == c)
            if len(coord) != 0:
                for item in coord:
                    swir_candi = np.logical_and(raw_img[:, :, 2] > (h - item[0] - 1) * h_interval * 10000,
                                                raw_img[:, :, 2] < (h - item[0]) * h_interval * 10000)
                    rded_candi = np.logical_and(raw_img[:, :, 0] > item[1] * w_interval * 10000,
                                                raw_img[:, :, 0] < (item[1] + 1) * w_interval * 10000)
                    pred[idx_b, swir_candi * rded_candi] = c

        if use_cm:
            target_copy = np.zeros_like(target)
            target_copy[target == 17] = 1
            target_copy[target == 98] = 2
            target_copy[target == 99] = 2
            target_copy[target == 147] = 3

            cm += confusion_matrix(pred, target_copy, classes)
            return cm, pred, target

def save_fig(pred_hist,
             label,
             image,
             pred_img,
             target_img,
             save_dir,
             fig_name,
             mode='single',
             title=None):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from matplotlib import cm
    fig1 = plt.figure(1, figsize=(14, 14))
    if mode == 'all':
        # plt.subplot(2, 2, 1)
        # plt.imshow(make_grid(pred_hist, nrow=8).cpu().numpy().transpose(1, 2, 0))
        # plt.subplot(2, 2, 2)
        # plt.imshow(make_grid(target_hist, nrow=8).cpu().numpy().transpose(1, 2, 0))
        gist_ncar_r = cm.get_cmap('gist_ncar_r', 256)
        cmap1 = gist_ncar_r(np.arange(0, 256))
        cmap1[:1, :] = np.array([1, 1, 1, 1])
        cmap1 = ListedColormap(cmap1)
        if title:
            plt.rcParams["figure.titlesize"] = 'large'
            plt.rcParams["figure.titleweight"] = 'bold'
            plt.rcParams["font.family"] = 'Arial'
            plt.rcParams["font.size"] = '18'
            plt.suptitle(title)

        plt.subplot(2, 2, 1)
        plt.imshow(make_grid(image, nrow=2, padding=10).cpu().numpy().transpose(1, 2, 0)[:, :, 0], cmap=cmap1)
        plt.xticks([0, 31, 63, 95, 127], np.round(np.linspace(0, 0.3, 5), 4))
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')

        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

        plt.subplot(2, 2, 2)
        c_candi = np.array([[255, 255, 255, 255],
                            [255, 211, 0, 255],
                            [38, 112, 0, 255],
                            [168, 0, 229, 255]
                            ]) / 255.0
        pred_hist_c = np.unique(pred_hist.cpu().numpy().squeeze()).astype(np.int)
        cmap2 = ListedColormap(c_candi[pred_hist_c, :])
        plt.imshow(pred_hist.cpu().numpy().squeeze(), cmap=cmap2)
        plt.xticks([0, 31, 63, 95, 127], np.round(np.linspace(0, 0.3, 5), 4))
        plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')

        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

        # plt.subplot(2, 3, 3)
        # for i in range(label.shape[2]):
        #     min_v = label[:, :, i].min()
        #     max_v = label[:, :, i].max()
        #     label[:, :, i] = (label[:, :, i] - min_v) / (max_v - min_v) * 255
        # plt.imshow(label)
        # plt.xticks([0, 31, 63, 95, 127], [0, 0.075, 0.15, 0.225, 0.3])
        # plt.yticks([0, 31, 63, 95, 127], [0.6, 0.45, 0.3, 0.15, 0])

        plt.subplot(2, 2, 3)
        plt.imshow(pred_img.squeeze(), cmap=cmap2)
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')
        # plt.yticks([12, 112, 212, 312, 412, 512], [500, 400, 300, 200, 100, 0])
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

        plt.subplot(2, 2, 4)
        cmap3 = ListedColormap(np.array([[255, 255, 255, 255],
                                         [255, 211, 0, 255],
                                         [38, 112, 0, 255],
                                         [168, 0, 229, 255]
                                         ]) / 255.0)
        plt.imshow(np.array(target_img).squeeze(), cmap=cmap3)
        # plt.yticks([12, 112, 212, 312, 412, 512], [500, 400, 300, 200, 100, 0])
        ax = plt.gca()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            label.set_fontname('Arial')
            # label.set_style('italic')
            label.set_fontsize(16)
            label.set_weight('bold')
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)

    if mode == 'single':
        plt.imshow(pred_img.squeeze())
    # plt.grid()
    # plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name))

def writeTif(bands, path, require_proj=False, transform=None, proj=None):
    if bands is None or bands.__len__() == 0:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            if require_proj:
                dataset.SetGeoTransform(transform)  # 写入仿射变换参数
                dataset.SetProjection(proj)  # 写入投影
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")

def create_map_all_nonoverlap(pred_dir, CDl_dir, result_filename, class_index=[1, 5], class_name=['corn', 'soybean']):
    x_range = 257
    y_range = 257
    radius = (x_range - 1) // 2

    img = gdal.Open(CDl_dir, GA_ReadOnly)
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()

    top_left_x = img_geotrans[0]
    w_e_pixel_resolution = img_geotrans[1]
    top_left_y = img_geotrans[3]
    n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range

    x_size = x_num * x_range
    y_size = y_num * y_range
    x_off = 0
    y_off = 0
    cdl = img.ReadAsArray(0, 0, x_size, y_size)
    target = np.zeros_like(cdl)
    target[cdl == 17] = 1
    target[cdl == 98] = 2
    target[cdl == 99] = 2
    target[cdl == 147] = 3

    class_sum = [(cdl == i).sum() for i in class_index]

    columns = ['time index']
    for c in class_name:
        columns.append(c + '_number')
        columns.append(c + '_precision')
        columns.append(c + '_recall')
    if not os.path.exists(os.path.join(root_dir, result_filename + '.csv')):
        df = pd.DataFrame(data=[], columns=columns)
        df.to_csv(os.path.join(root_dir, result_filename + '.csv'), index=False)
    time_index = []

    for ts in range(0, 24):
        # if ts == 16:
        #     continue
        weight_map = np.zeros([4, y_size, x_size])
        time_index.append(ts)
        # time_index = [ts]
        for pred in os.listdir(pred_dir):
            if pred.endswith('.tif'):
                if int(pred.split('.')[0].split('_')[0]) not in time_index:
                    continue
                img = io.imread(os.path.join(pred_dir, pred))
                index = int(pred.split('.')[0].split('_')[1])
                col = index // y_num
                row = index % y_num
                # pred_collaged[crop_index, row * y_range : (row + 1) * y_range , col * x_range : (col + 1) * x_range ] = img
                for c in range(1, len(class_index) + 1):
                    test1 = -row * y_range + (row + 1) * y_range
                    test2 = -col * x_range + (col + 1) * x_range
                    test = weight_map[c-1, row * y_range: (row + 1) * y_range, col * x_range: (col + 1) * x_range]

                    weight_map[c-1, row * y_range: (row + 1) * y_range, col * x_range: (col + 1) * x_range] += (
                                    img == c).astype(np.int)
                # if img.sum() !=0:
                #     corn_sum += (cdl[row * y_range : (row + 1) * y_range , col * x_range : (col + 1) * x_range]==1).sum()
                #     soybean_sum += (cdl[row * y_range: (row + 1) * y_range, col * x_range: (col + 1) * x_range] == 5).sum()

        background = (np.sum(weight_map, axis=0) == 0)
        wheat = (np.argmax(weight_map, axis=0) == 0) #* (weight_map[0, :, :] > 1)
        corn = (np.argmax(weight_map, axis=0) == 1) #* (weight_map[1, :, :] > 1)
        sunflower = (np.argmax(weight_map, axis=0) == 2) * (weight_map[2, :, :] > 1)
        # corn = (pred_collaged[0] == 1) * (pred_collaged[1] == 0)
        # soybean = (pred_collaged[0] == 0) * (pred_collaged[1] == 1)
        classification = np.zeros([y_size, x_size])

        classification[wheat] = 0
        classification[corn] = 1
        classification[sunflower] = 2
        classification[background] = 3
        data = [time_index]
        for i in range(0, len(class_index)):
            # print((classification == i ).sum())
            # print(((classification == i ) * (cdl == class_index[i])).sum() / (classification == i).sum())
            # print(((classification == i ) * (cdl == class_index[i])).sum() / class_sum[i])
            try:
                data.append((classification == i ).sum())
                data.append(((classification == i ) * (target == class_index[i])).sum() / (classification == i).sum())
                data.append(((classification == i ) * (target == class_index[i])).sum() / class_sum[i])
            except:
                pass
        df = pd.read_csv(os.path.join(root_dir, result_filename + '.csv'))
        df = df.append(pd.DataFrame.from_dict(
            data={0: data},
            orient='index', columns=columns), ignore_index=True)
        df.to_csv(os.path.join(root_dir, result_filename + '.csv'), index=False)
        # new_top_left_x = top_left_x + x_off * np.abs(w_e_pixel_resolution)
        # new_top_left_y = top_left_y - y_off * np.abs(n_s_pixel_resolution)
        #
        # dst_transform = (
        #     new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4],
        #     img_geotrans[5])
        # driver = gdal.GetDriverByName("GTiff")
        # path = os.path.join(root_dir+r'\classification1', r'2019_sample_accu_' + str(ts)+'.tif')
        # dataset = driver.Create(path, classification.shape[1], classification.shape[0], 1, gdal.GDT_Float32)
        # if dataset is not None:
        #     dataset.SetGeoTransform(dst_transform)  # 写入仿射变换参数
        #     dataset.SetProjection(img_proj)  # 写入投影
        #     for i in range(1):
        #         dataset.GetRasterBand(i + 1).WriteArray(classification[:, :])

def create_map_all(pred_dir, CDl_dir, result_filename, class_index=[1, 5], class_name=['corn', 'soybean']):
    x_range = 257
    y_range = 257
    radius = (x_range - 1) // 2

    img = gdal.Open(CDl_dir, GA_ReadOnly)
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()

    top_left_x = img_geotrans[0]
    w_e_pixel_resolution = img_geotrans[1]
    top_left_y = img_geotrans[3]
    n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range

    x_size = x_num * x_range
    y_size = y_num * y_range
    x_off = 0
    y_off = 0
    cdl = img.ReadAsArray(0, 0, x_size, y_size)
    target = np.zeros_like(cdl)
    target[cdl == 17] = 1
    target[cdl == 98] = 2
    target[cdl == 99] = 2
    target[cdl == 147] = 3
    class_sum = [(target == i).sum() for i in class_index]
    columns = ['time index']
    for c in class_name:
        columns.append(c + '_number')
        columns.append(c + '_precision')
        columns.append(c + '_recall')
    if not os.path.exists(os.path.join(root_dir, result_filename + '.csv')):
        df = pd.DataFrame(data=[], columns=columns)

        df.to_csv(os.path.join(root_dir, result_filename + '.csv'), index=False)
    time_index = []
    for ts in range(0, 24):
        # if ts == 16:
        #     continue
        weight_map = np.zeros([len(class_index), y_size, x_size])
        time_index.append(ts)
        # time_index = [ts]
        for pred in os.listdir(pred_dir):
            if pred.endswith('.tif'):
                if int(pred.split('.')[0].split('_')[0]) not in time_index:
                    continue
                # if pred != '10_4138_2086.tif':
                #     continue
                img = io.imread(os.path.join(pred_dir, pred))
                x_coor = int(pred.split('.')[0].split('_')[1])
                y_coor = int(pred.split('.')[0].split('_')[2])
                try:
                    for c in range(1, len(class_index) + 1):
                        weight_map[c - 1, y_coor - radius:y_coor + radius + 1, x_coor - radius:x_coor + radius + 1] += (
                                    img == c).astype(np.int)
                except:
                    pass

        background = (np.sum(weight_map, axis=0) == 0)
        wheat = (np.argmax(weight_map, axis=0) == 0) #* (weight_map[0, :, :] > 1)
        corn = (np.argmax(weight_map, axis=0) == 1) #* (weight_map[1, :, :] > 1)
        sunflower = (np.argmax(weight_map, axis=0) == 2) * (weight_map[2, :, :] > 1)
        # corn = (pred_collaged[0] == 1) * (pred_collaged[1] == 0)
        # soybean = (pred_collaged[0] == 0) * (pred_collaged[1] == 1)
        classification = np.zeros([y_size, x_size])

        classification[wheat] = 1
        classification[corn] = 2
        classification[sunflower] = 3
        classification[background] = 0
        data = [time_index]
        for i in range(0, len(class_index)):
            # print((classification == i+1).sum())
            # print(((classification == i+1) * (cdl == class_index[i])).sum() / (classification == i+1).sum())
            # print(((classification == i+1) * (cdl == class_index[i])).sum() / class_sum[i])
            data.append((classification == i+1).sum())
            data.append(((classification == i+1) * (target == class_index[i])).sum() / (classification == i+1).sum())
            data.append(((classification == i+1) * (target == class_index[i])).sum() / class_sum[i])
        df = pd.read_csv(os.path.join(root_dir, result_filename + '.csv'))
        df = df.append(pd.DataFrame.from_dict(
            data={0: data},
            orient='index', columns=columns), ignore_index=True)
        df.to_csv(os.path.join(root_dir, result_filename+'.csv'), index=False)
        # new_top_left_x = top_left_x + x_off * np.abs(w_e_pixel_resolution)
        # new_top_left_y = top_left_y - y_off * np.abs(n_s_pixel_resolution)
        #
        # dst_transform = (
        #     new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4],
        #     img_geotrans[5])
        # driver = gdal.GetDriverByName("GTiff")
        # path = os.path.join(root_dir, r'2020_sample_' + str(ts)+'.tif')
        # dataset = driver.Create(path, classification.shape[1], classification.shape[0], 1, gdal.GDT_Float32)
        # if dataset is not None:
        #     dataset.SetGeoTransform(dst_transform)  # 写入仿射变换参数
        #     dataset.SetProjection(img_proj)  # 写入投影
        #     for i in range(1):
        #         dataset.GetRasterBand(i + 1).WriteArray(classification[:, :])

def creat_map_mosaic(dir1, dir2, CDl_dir):
    cdl = gdal.Open(CDl_dir, GA_ReadOnly)
    img_geotrans = cdl.GetGeoTransform()
    img_proj = cdl.GetProjection()
    top_left_x = img_geotrans[0]
    w_e_pixel_resolution = img_geotrans[1]
    top_left_y = img_geotrans[3]
    n_s_pixel_resolution = img_geotrans[5]
    x_off = 0
    y_off = 0
    for ts in range(0, 24):
        for pred in os.listdir(dir1):
            if pred.endswith('.tif'):
                if int(pred.split('.')[0].split('_')[-1]) != ts:
                    continue
                img1 = io.imread(os.path.join(dir1, pred)) ## rice/corn
                img2 = io.imread(os.path.join(dir2, pred)) ## corn/soybean
                img = np.ones_like(img1) * 3
                img[img1 == 0] = 0
                img[img2 == 2] = 2
                img[img1 == 1] = 1
                img[img2 == 1] = 1
                new_top_left_x = top_left_x + x_off * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - y_off * np.abs(n_s_pixel_resolution)

                dst_transform = (
                    new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4],
                    img_geotrans[5])
                driver = gdal.GetDriverByName("GTiff")
                path = os.path.join(root_dir + r'\classification', r'2019_sample_accu_' + str(ts) + '.tif')
                dataset = driver.Create(path, img.shape[1], img.shape[0], 1, gdal.GDT_Float32)
                if dataset is not None:
                    dataset.SetGeoTransform(dst_transform)  # 写入仿射变换参数
                    dataset.SetProjection(img_proj)  # 写入投影
                    for i in range(1):
                        dataset.GetRasterBand(i + 1).WriteArray(img[:, :])


if __name__ == '__main__':
    root_dir = r'F:\DigitalAG\liheng\EU\wt_sfl_cn'
    # create_map(os.path.join(root_dir, r'result_corn_2018'), os.path.join(root_dir, r'result_soybean_2018'),
    #            os.path.join(root_dir, 'IW_CDL_2018.tif'), 'sample_map_accuracy_accu_2018')
    # create_map_all(os.path.join(root_dir, r'result'),
    #                os.path.join(root_dir, r'CDL_2019.tif'),
    #                'sample_map_accuracy_accu_2019',
    #                [1, 2, 3], ['wheat', 'corn', 'sunflower'])
    # creat_map_mosaic(os.path.join(root_dir, 'classification1'),
    #                  os.path.join(root_dir, 'classification2'),
    #                  os.path.join(root_dir, r'2019\raw data\CDL_2019.tif'))


