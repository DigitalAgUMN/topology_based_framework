#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chenxi
"""
import torch.utils as utils
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utility_functions import get_hist2d, get_target, get_separability, confusion_matrix_from_hist, plot_hist2d
from utility_functions import get_coordinate
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid



def scale_percentile_n(matrix):
    # matrix = matrix.transpose(1, 2, 0)
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)  # 百分位的值是在一个列表中排序 再取
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 0, axis=0)  # 取第2百分位的值 储存在mins中
    # print(mins,"-----",mins[None:])
    maxs = np.percentile(matrix, 100, axis=0)  # - mins
    # print(maxs,"-----",maxs[None:])
    matrix = (matrix - mins[None:]) / (maxs[None:] * 0.5) - 1
    matrix = np.reshape(matrix, [w, h, d]).astype(np.float32)
    matrix = matrix.clip(-1, 1)
    # print(matrix)
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

def generate_hist(dir, folder_id='', file_id='', indexes=[9,16], positive=True, test=False):
    ## create folders
    for item in ['img', 'target', 'RGB']:
        if not os.path.exists(os.path.join(root_dir, folder_id+ item)):
            os.makedirs(os.path.join(root_dir, folder_id + item))

    img_dir = np.array(pd.read_csv(dir)['img_dir'])
    target_dir = np.array(pd.read_csv(dir)['target_dir'])
    np.random.seed(2000)
    dirs = np.c_[img_dir, target_dir]
    np.random.shuffle(dirs)
    zeros_count = 0

    for index, img_target in enumerate(dirs):
        patch_index = img_target[0].split('.')[:-1][0].split('\\')[-1]
        date_index = int(img_target[0].split('.')[:-1][0].split('\\')[-2].split('_')[0])
        # if date_index != 5 or patch_index != '105':
        #     continue
        if not test:
            if positive:
                if date_index >= indexes[0] and date_index <= indexes[1]:
                    img = io.imread(img_target[0])[:,:,[3, 8]]/ 10000
                    if (img != 0).mean() < 0.8:
                        continue
                    target = io.imread(img_target[1])
                    mask = io.imread(r'F:\DigitalAG\liheng\MN\2017\segmentation\target_historical\\' + img_target[1].split('\\')[-1])
                    # make sure the shape is (channel, height, weight)
                    # each image is a square
                    if img.shape[0] == img.shape[1]:
                        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]

                    """
                    convert the image to 3 classes:
                    0 for background
                    1 for corn
                    2 for soybean
                    """

                    code_class = [1, 5, 41]
                    if (target == 1).sum() == 0:
                        data_target == np.zeros([4,128,128])
                    else:
                        arr_class = np.zeros_like(target)
                        for i_cur, code_cur in enumerate(code_class):
                            arr_class[target == code_cur] = i_cur + 1
                        list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
                        mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
                        sep = get_separability(img, arr_class)
                        if (sep > 0.9).sum()==6:
                            list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                            x_coor = get_coordinate(list_img2)
                            y_coor = get_coordinate(list_img2, x_coor=False)

                            data_target = get_target\
                            (
                                list_img,
                                separability=sep,
                                percentile_pure=65,
                                threshold_separability=0.9,
                                crop='all'
                            )
                            name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                                      np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                                      x_coor[np.newaxis, np.newaxis, :,:],
                                                      y_coor[np.newaxis, np.newaxis, :,:]), axis=1).squeeze().astype(np.int16))
                            if (mask==0).sum() <= mask.shape[0] ** 2 * 0.5:
                                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                          data_target.squeeze().astype(np.int8))
                            if (mask==0).sum() > mask.shape[0] ** 2 * 0.5:
                                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                          np.zeros([4, 128, 128]).astype(np.int8))
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                                      np.array(list_img).squeeze().astype(np.int16))
                            zeros_count += 1
                            if zeros_count > 1000:
                                break


            if not positive:
                img = io.imread(img_target[0]) / 10000
                if (img != 0).mean() < 0.8:
                    continue
                target = io.imread(img_target[1])
                mask = io.imread(
                    r'F:\DigitalAG\liheng\MN\2017\segmentation\target_historical\\' + img_target[1].split('\\')[-1])
                # make sure the shape is (channel, height, weight)
                # each image is a square
                if img.shape[0] == img.shape[1]:
                    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
                code_class = [1, 5, 41]
                if (target == 1).sum() == 0:
                    data_target == np.zeros([1, 1, 128, 128])
                else:
                    arr_class = np.zeros_like(target)
                    for i_cur, code_cur in enumerate(code_class):
                        arr_class[target == code_cur] = i_cur + 1
                    list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
                    mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
                    sep = get_separability(img, arr_class)

                    if date_index < indexes[0] or date_index > indexes[1]:
                        list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                        x_coor = get_coordinate(list_img2)
                        y_coor = get_coordinate(list_img2, x_coor=False)
                        name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                                  np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                                  x_coor[np.newaxis, np.newaxis, :, :],
                                                  y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                                      np.int16))
                        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                                  np.array(list_img).squeeze().astype(np.int16))
                        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                  np.zeros([4, 128, 128]).astype(np.int8))
                        zeros_count += 1

                    else:
                        if sep.mean() < 0.9:
                            list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                            x_coor = get_coordinate(list_img2)
                            y_coor = get_coordinate(list_img2, x_coor=False)
                            data_target = get_target(
                                list_img,
                                separability=sep,
                                percentile_pure=0.65,
                                threshold_separability=0.9,
                                crop='all'
                            )
                            name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                                      np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                                      x_coor[np.newaxis, np.newaxis, :, :],
                                                      y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                                          np.int16))
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                                      data_target.squeeze().astype(np.int8))
                            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                                      np.array(list_img).squeeze().astype(np.int16))
                            zeros_count += 1
                    if zeros_count > 1000:
                        break

        if test:
            candidate = np.zeros(68)
            candidate[0] = 660
            for i_c in range(1, 68):
                if i_c % 4 != 0:
                    candidate[i_c] = candidate[i_c - 1] + 1
                if i_c % 4 == 0:
                    candidate[i_c] = candidate[i_c - 1] + 19
            # if int(patch_index) in candidate:
            img = io.imread(img_target[0]) / 10000
            target = io.imread(img_target[1])
            mask = io.imread(
                r'E:\DigitalAG\liheng\IW\target_historical\\' + img_target[1].split('\\')[-1])
            if img.shape[0] == img.shape[1]:
                img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
            code_class = [1, 5]
            if (target == 1).sum() == 0:
                data_target == np.zeros([1, 1, 128, 128])
            else:
                arr_class = np.zeros_like(target)
                for i_cur, code_cur in enumerate(code_class):
                    arr_class[target == code_cur] = i_cur + 1
                list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
                list_img2, list_xedges2, list_yedges2 = get_hist2d(img, bins_range=bins_range)
                x_coor = get_coordinate(list_img2)
                y_coor = get_coordinate(list_img2, x_coor=False)
                mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8), bins_range=bins_range)
                name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                          np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                          x_coor[np.newaxis, np.newaxis, :, :],
                                          y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(np.int16))
                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                          np.array(list_img).squeeze().astype(np.int16))
                # io.imsave(os.path.join(os.path.join(root_dir, folder_id + '_target'), name),
                #           data_target.squeeze().astype(np.int8))

def heat_map_pos_training(dir, folder_id='', file_id='', year=2019):
    ## create folders
    for item in ['img', 'target', 'RGB', 'JM']:
        if not os.path.exists(os.path.join(root_dir, folder_id + item)):
            os.makedirs(os.path.join(root_dir, folder_id + item))
    img_dir = np.array(pd.read_csv(dir)['img_dir'])
    target_dir = np.array(pd.read_csv(dir)['target_dir'])
    np.random.seed(2000)
    dirs = np.c_[img_dir, target_dir]
    np.random.shuffle(dirs)
    count = 0

    for index, img_target in enumerate(dirs):
        patch_index = img_target[0].split('.')[:-1][0].split('\\')[-1]
        date_index = int(img_target[0].split('.')[:-1][0].split('\\')[-2].split('_')[0])
        # img = io.imread(img_target[0])[:, :, [3, 8]] / 10000
        img = io.imread(img_target[0])[:, :, [1, 2]] / 10000
        # if (img != 0).mean() < 0.9:
        #     continue
        target = io.imread(img_target[1])
        mask = io.imread(img_target[1].replace('target', 'target_historical'))
        theia = io.imread(img_target[1].replace('target', 'mask')).astype(np.int)
        # make sure the shape is (channel, height, weight)
        # each image is a square
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        code_class = [1, 3, 4]
        arr_class = np.zeros_like(target)
        for i_cur, code_cur in enumerate(code_class):
            arr_class[target == code_cur] = i_cur + 1
        list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
        mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8),
                                                        bins_range=bins_range)
        try:
            sep = get_separability(img, arr_class)
        except:
            continue
        if (sep > 0.5).sum() == len(sep.flatten()):
            list_img2, list_xedges2, list_yedges2 = get_hist2d(img, mask=theia, bins_range=bins_range)
            x_coor = get_coordinate(list_img2)
            y_coor = get_coordinate(list_img2, x_coor=False)
            data_target = get_target \
                    (
                    list_img,
                    separability=sep,
                    percentile_pure=50,
                    threshold_separability=0.5,
                    crop='all'
                )
            name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
            np.save(os.path.join(root_dir, folder_id + 'JM', name.replace('.tif', '.npy')), sep[0])
            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                      np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                      x_coor[np.newaxis, np.newaxis, :, :],
                                      y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                          np.int16))
            if (mask == -1).sum() <= mask.shape[0] ** 2 * 0.5:
                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                          data_target.squeeze().astype(np.int8))
            if (mask == -1).sum() > mask.shape[0] ** 2 * 0.5:
                io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                          np.zeros([len(code_class) + 1, 128, 128]).astype(np.int8))
            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                      np.array(list_img).squeeze().astype(np.int16))

def heat_map_neg_training(dir, folder_id='', file_id='', year=2019):
    ## create folders
    for item in ['img', 'target', 'RGB', 'JM']:
        if not os.path.exists(os.path.join(root_dir, folder_id + item)):
            os.makedirs(os.path.join(root_dir, folder_id + item))
    img_dir = np.array(pd.read_csv(dir)['img_dir'])
    target_dir = np.array(pd.read_csv(dir)['target_dir'])
    np.random.seed(2000)
    dirs = np.c_[img_dir, target_dir]
    np.random.shuffle(dirs)
    count = 0

    for index, img_target in enumerate(dirs):
        patch_index = img_target[0].split('.')[:-1][0].split('\\')[-1]
        date_index = int(img_target[0].split('.')[:-1][0].split('\\')[-2].split('_')[0])
        # img = io.imread(img_target[0])[:, :, [3, 8]] / 10000
        img = io.imread(img_target[0])[:, :, [1, 2]]  / 10000
        if (img != 0).mean() < 0.9:
            continue
        target = io.imread(img_target[1])
        code_class = [1, 3, 4]
        mask = io.imread(img_target[1].replace('target', 'target_historical'))
        theia = io.imread(img_target[1].replace('target', 'mask')).astype(np.int)

        # make sure the shape is (channel, height, weight)
        # each image is a square
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        arr_class = np.zeros_like(target)
        for i_cur, code_cur in enumerate(code_class):
            arr_class[target == code_cur] = i_cur + 1
        list_img, list_xedges, list_yedges = get_hist2d(img, arr_class=arr_class, bins_range=bins_range)
        mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8),
                                                             bins_range=bins_range)
        try:
            sep = get_separability(img, arr_class)
        except:
            continue
        if (sep > 0.5).sum() != len(sep.flatten()):
            list_img2, list_xedges2, list_yedges2 = get_hist2d(img, mask=theia, bins_range=bins_range)
            x_coor = get_coordinate(list_img2)
            y_coor = get_coordinate(list_img2, x_coor=False)
            data_target = get_target \
                    (
                    list_img,
                    separability=sep,
                    percentile_pure=50,
                    threshold_separability=1,
                    crop='all'
                )
            name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
            np.save(os.path.join(root_dir, folder_id + 'JM', name.replace('.tif', '.npy')), sep[0])
            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                      np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                      x_coor[np.newaxis, np.newaxis, :, :],
                                      y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                          np.int16))
            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                          np.zeros([len(code_class) + 1, 128, 128]).astype(np.int8))
            io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'RGB'), name),
                      np.array(list_img).squeeze().astype(np.int16))
            count += 1
            if count > 300:
                break

def heat_map_testing(dir, folder_id='', file_id=''):
    ## create folders
    for item in ['img', 'target', 'RGB']:
        if not os.path.exists(os.path.join(root_dir, folder_id + item)):
            os.makedirs(os.path.join(root_dir, folder_id + item))
    img_dir = np.array(pd.read_csv(dir)['img_dir'])
    target_dir = np.array(pd.read_csv(dir)['target_dir'])
    np.random.seed(2000)
    dirs = np.c_[img_dir, target_dir]
    np.random.shuffle(dirs)
    count = 0

    for index, img_target in enumerate(dirs):
        patch_index = img_target[0].split('.')[:-1][0].split('\\')[-1]
        date_index = int(img_target[0].split('.')[:-1][0].split('\\')[-2].split('_')[0])
        img = io.imread(img_target[0])[:, :, [0, 2]] / 10000
        # if (img != 0).mean() < 0.9:
        #     continue
        mask = io.imread(img_target[1].replace('target', 'target_historical'))
        theia = io.imread(img_target[1].replace('target', 'mask')).astype(np.int)
        # make sure the shape is (channel, height, weight)
        # each image is a square
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        code_class = [1, 2, 5]
        mask_img, mask_xedges, mask_yedges = get_hist2d(img, arr_class=(mask != 0).astype(np.int8),
                                                            bins_range=bins_range)

        list_img2, list_xedges2, list_yedges2 = get_hist2d(img, mask=theia, bins_range=bins_range)
        x_coor = get_coordinate(list_img2)
        y_coor = get_coordinate(list_img2, x_coor=False)
        name = str(date_index) + '_' + str(patch_index) + file_id + '.tif'
        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'img'), name),
                  np.concatenate((list_img2, mask_img[:, 0:1, :, :],
                                  x_coor[np.newaxis, np.newaxis, :, :],
                                  y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                      np.int16))
        io.imsave(os.path.join(os.path.join(root_dir, folder_id + 'target'), name),
                      np.zeros([len(code_class) + 1, 128, 128]).astype(np.int8))

def get_class_weight(dir, class_num=2):
    target_list = os.listdir(dir)
    class_pixels = np.zeros(class_num)
    for item in target_list:
        try:
            target = io.imread(os.path.join(dir, item))
        except:
            pass
        target[target!=0]=1
        for i in range(0, class_num):
            class_pixels[i] += (target == i).sum()
    weight = class_pixels / (len(target_list) * 128 * 128) * 100
    return 1 / (weight / weight[0])

def remove_file(dir):
    [os.remove(os.path.join(dir, file)) for file in os.listdir(dir)]

def match_folders(f1_dir, f2_dir, identifier='', rename=False):
    f1_list = np.array(os.listdir(f1_dir))
    f2_list = np.array(os.listdir(f2_dir))
    union = np.intersect1d(f1_list, f2_list)
    diff_f1 = np.setdiff1d(f1_list, union)
    diff_f2 = np.setdiff1d(f2_list, union)
    [os.remove(os.path.join(f1_dir, file)) for file in diff_f1]
    [os.remove(os.path.join(f2_dir, file)) for file in diff_f2]
    # if rename:
    #     [os.rename(os.path.join(f1_dir, file), os.path.join(f1_dir,
    #                                                         file.replace(file.split('.')[0], file.split('.')[0]+identifier))) for file in os.listdir(f1_dir)]
    #     [os.rename(os.path.join(f2_dir, file), os.path.join(f2_dir,
    #                                                         file.replace(file.split('.')[0], file.split('.')[0]+identifier))) for file in os.listdir(f2_dir)]

def check_data_target_RGB(data_dir, identifier='training_ls7', sensor='landsat'):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    if sensor == 'landsat':
        xticks = {'raw':[0, 31, 63, 95, 127], 'scale': [0, 0.2, 0.4, 0.6, 0.8]}
        yticks = {'raw': [0, 31, 63, 95, 127], 'scale': [0.6, 0.45, 0.3, 0.15, 0]}
    else:
        xticks = {'raw': [0, 31, 63, 95, 127], 'scale': [round(i, 1) for i in np.linspace(0, bins_range[0], 5)]}
        yticks = {'raw': [0, 31, 63, 95, 127], 'scale': [round(i, 1) for i in np.linspace(bins_range[1], 0, 5)]}
    img_dir = os.path.join(data_dir, identifier+'_img')
    for dir in os.listdir(img_dir):
        img = io.imread(os.path.join(img_dir, dir))
        target = io.imread(os.path.join(img_dir.replace('img', 'target'), dir))
        JM = np.load(os.path.join(img_dir.replace('img', 'JM'), dir.replace('.tif', '.npy')))
        fig = plt.figure(1, figsize=[12,6])
        title = ''
        for i in range(len(JM)):
            title += str(round(JM[i], 4)) + '/'
        title = title[:-1]
        plt.title(title, fontdict={'family': 'Arial', 'size': 12, 'weight': 'bold'}, x=-1.5, y=1.2)
        if len(target.shape) == 3 and target.shape[0] != 128:
            target = target.transpose(1,2,0)
        rgb = io.imread(os.path.join(img_dir.replace('img', 'RGB'), dir))
        fig = plt.figure(1, figsize=(14, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(img[:,:, 0], cmap='gist_ncar_r')
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        plt.subplot(1, 4, 2)
        plt.imshow(img[:,:, 1], cmap='gist_ncar_r')
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        plt.subplot(1, 4, 3)
        color_list = np.array([[255, 255, 255],
                                        [255, 211, 0],
                                        [38, 112, 0],
                                        [168, 0, 229],
                                        [255, 165, 226]
                                        ]) / 255.0
        cmap = ListedColormap(color_list[:target.shape[2]])
        plt.imshow(target.sum(axis=2), cmap=cmap)
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        plt.subplot(1, 4, 4)
        for i in range(rgb.shape[2]):
            min_v = rgb[:, :, i].min()
            max_v = rgb[:, :, i].max()
            rgb[:, :, i] = (rgb[:, :, i] - min_v) / (max_v - min_v) * 255
        plt.imshow(rgb[:, :, 1:4])
        plt.xticks(xticks['raw'], xticks['scale'])
        plt.yticks(yticks['raw'], yticks['scale'])
        folder_name = r'visualize_' + identifier
        if not os.path.exists(os.path.join(root_dir, folder_name)):
            os.makedirs(os.path.join(root_dir, folder_name))
        plt.savefig(os.path.join(os.path.join(root_dir, folder_name), dir))
        plt.cla()

def combine_folders(src, dst, identifier=''):
    import shutil
    for file in os.listdir(src):
        shutil.copyfile(os.path.join(src, file), os.path.join(dst, identifier+file))


global root_dir, bins_range
root_dir = r'F:\DigitalAG\liheng\EU\2018\wt_sfl_cn\model data(nir)'
bins_range = [0.8, 0.6]
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

if __name__ == '__main__':
    operations = {'remove': False, 'match': False, 'combine': False, 'generate': False}
    op = [4]
    for index, key in enumerate(operations.keys()):
        if index + 1 in op:
            operations[key] = True

    if operations['remove']:
        remove_file(os.path.join(root_dir, 'training_pos_img'))
        remove_file(os.path.join(root_dir, 'training_pos_target'))

    elif operations['match']:
        for type in ['pos', 'neg']:
            match_folders(os.path.join(root_dir, r'training_'+type+'_img'),
                          os.path.join(root_dir, r'visualize_training_' + type))
            match_folders(os.path.join(root_dir, r'training_'+type+'_target'),
                          os.path.join(root_dir, r'training_'+type+'_img'))

    elif operations['combine']:
        for type in ['img', 'target']:
            folder_names = ['training_pos_' + type, 'training_neg_' + type]
            # folder_names = ['training_18_negative2_' + type]
            for folder in folder_names:
                src = os.path.join(os.path.join(r'F:\DigitalAG\liheng\EU\2018\wt_sfl_cn\model data(nir)\\' + folder))
                dst = r'F:\DigitalAG\liheng\EU\wt_sfl_cn\training_' + type
                combine_folders(src, dst, '')

    elif operations['generate']:
        sensor = ''
        # heat_map_pos_training(r'F:\DigitalAG\liheng\EU\2018\wt_sfl_cn\segmentation\training_grid.csv', folder_id='training_pos_'+sensor, year=2018)
        # check_data_target_RGB(root_dir, identifier='training_pos' + sensor, sensor='sentinel-2')
        heat_map_neg_training(r'F:\DigitalAG\liheng\EU\2018\wt_sfl_cn\segmentation\training_grid.csv', folder_id='training_neg_' + sensor, year=2018)
        check_data_target_RGB(root_dir, identifier='training_neg'+sensor, sensor='sentinel-2')
        # heat_map_testing(r'F:\DigitalAG\liheng\EU\2019_early\wt_bt_rp\segmentation\training_grid.csv', folder_id='testing_'+sensor)

