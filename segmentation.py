#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from osgeo import gdal
from gdalconst import *
import os
import pandas as pd
from skimage import io


# In[2]:


def non_overlap_segmentation(img, folder, x_range, y_range, target=False, require_proj=False, continue_count=False):
    if require_proj:
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        top_left_x = img_geotrans[0]
        w_e_pixel_resolution = img_geotrans[1]
        top_left_y = img_geotrans[3]
        n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)

    if continue_count:
        original_count = len(os.listdir(os.path.join(folder, 'input')))
    else:
        original_count = 0

    for i in range(0, x_num):
        for j in range(0, y_num):
            x_off_patch = i * x_range
            y_off_patch = j * y_range
            if not target:
                patch = img_array[:, y_off_patch:y_off_patch + y_range, x_off_patch:x_off_patch + x_range]
                ## determine if the patch has enough valid pixel
                valid_ratio = (patch != 0).mean()
                if valid_ratio < 0.8:
                    continue

            if target:
                patch = img_array[y_off_patch:y_off_patch + y_range, x_off_patch:x_off_patch + x_range][np.newaxis, :,
                        :]

            patch_name = os.path.join(folder, str(i * y_num + j + original_count) + '.tif')

            if require_proj:
                new_top_left_x = top_left_x + x_off_patch * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - y_off_patch * np.abs(n_s_pixel_resolution)
                dst_transform = (
                    new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4], img_geotrans[5])
                writeTif(patch, patch_name, require_proj, dst_transform, img_proj)
            else:
                writeTif(patch, patch_name)

def random_segmentation(img, folder, x_range, y_range, target=False, require_proj=False, continue_count=False):
    if require_proj:
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        top_left_x = img_geotrans[0]
        w_e_pixel_resolution = img_geotrans[1]
        top_left_y = img_geotrans[3]
        n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)

    ## generate pixels coordinates
    np.random.seed(122)
    n = 10000
    radius = (x_range - 1) // 2
    x_coor = np.random.choice(range(radius, x_size - radius - 1), n)
    y_coor = np.random.choice(range(radius, y_size - radius - 1), n)
    filter = io.imread(os.path.join(root_dir, 'CDL_2018.tif'))


    for i in range(0, n):
        if not target:
            patch = img_array[:, y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
            ## determine if the patch has enough valid pixel
            valid_ratio = (patch != 0).mean()
            # if valid_ratio < 0.8:
            #     continue
            filter_patch = filter[y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
            if (filter_patch == 2).sum() < 0.2 * x_range * y_range:
                continue
            patch_name = os.path.join(folder, str(x_coor[i]) + '_' + str(y_coor[i]) + '.tif')
            writeTif(patch, patch_name)

        if target:
            patch = img_array[y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1][
                    np.newaxis, :,
                    :]
            filter_patch = filter[y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
            if (filter_patch == 2).sum() < 0.2 * x_range * y_range:

                continue

            new_top_left_x = top_left_x + (x_coor[i] - 128) * np.abs(w_e_pixel_resolution)
            new_top_left_y = top_left_y - (y_coor[i] - 128) * np.abs(n_s_pixel_resolution)

            dst_transform = (
                new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4],
                img_geotrans[5])
            patch_name = os.path.join(folder, str(x_coor[i]) + '_' + str(y_coor[i]) + '.tif')
            writeTif(patch, patch_name, require_proj=True, transform=dst_transform, proj=img_proj)

def writeTif(bands, path, require_proj=False, transform=None, proj=None):
    if bands is None or bands.__len__() == 0:
        return
    else:
        band1 = bands[0]
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            if require_proj:
                dataset.SetGeoTransform(transform)
                dataset.SetProjection(proj)
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")

def readImagePath(img_path, x_range, y_range, folder_name='', target=False, require_proj=True, continue_count=True):
    img = gdal.Open(img_path, GA_ReadOnly)
    if require_proj:
        img_geotrans = img.GetGeoTransform()  # crs transform information
        img_proj = img.GetProjection()  # projection
        top_left_x = img_geotrans[0]  # x coordinate of upper lefe corner
        w_e_pixel_resolution = img_geotrans[1]  # horizontal resolution
        top_left_y = img_geotrans[3]  # y coordinate of upper lefe corner
        n_s_pixel_resolution = img_geotrans[5]  # vertical resolution
    folder = os.path.join(save_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(folder + ' has been created')
    if not continue_count:
        [os.remove(os.path.join(folder, file)) for file in os.listdir(folder)]
    random_segmentation(img, folder, x_range, y_range, target, require_proj, continue_count=False)

def create_historical_CDL(img_list, target_class=[0]):
    target = []
    for t in img_list:
        img = gdal.Open(t, GA_ReadOnly)
        img_geotrans = img.GetGeoTransform()  # crs transform information
        img_proj = img.GetProjection()  # projection
        x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
        img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)
        target.append(img_array)
    historical_CDL = np.ones([y_size, x_size])
    for i in range(0, len(img_list)):
        for c in target_class:
            historical_CDL *= (target[i][:,:] != c)
    historical_CDL = -historical_CDL + 1
    writeTif(historical_CDL[np.newaxis,:,:], os.path.join(r'F:\DigitalAG\liheng\EU\CDL', 'historical_CDL_new.tif'), True, img_geotrans, img_proj)

def reclassify_CDL(dir, year='2019'):
    img = gdal.Open(dir, GA_ReadOnly)
    img_geotrans = img.GetGeoTransform()  # crs transform information
    img_proj = img.GetProjection()  # projection
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)
    img_new= np.zeros_like(img_array)
    if year == '2019':
        img_array[img_array == 99] = 98
        img_array[img_array == 148] = 147
        for idx, crop in enumerate([17, 18, 98, 147, 53, 123]):
            img_new[img_array == crop] = idx + 1
    elif year == '2018':
        img_array[img_array == 115] = 114
        img_array[img_array == 168] = 167
        for idx, crop in enumerate([19, 20, 114, 167, 57, 140]):
            img_new[img_array == crop] = idx + 1
    elif year == '2017':
        img_array[img_array == 111] = 110
        img_array[img_array == 164] = 163
        for idx, crop in enumerate([19, 20, 111, 163, 53, 134]):
            img_new[img_array == crop] = idx + 1
    elif year == '2016':
        img_array[img_array == 126] = 125
        img_array[img_array == 152] = 151
        for idx, crop in enumerate([18, 19, 125, 151, 52, 125]):
            img_new[img_array == crop] = idx + 1
    writeTif(img_new[np.newaxis, :, :], r'F:\DigitalAG\liheng\EU\CDL\CDL_' + year + '.tif', True, img_geotrans,
             img_proj)

def reclassify_theia(dir):
    img = gdal.Open(dir, GA_ReadOnly)
    img_geotrans = img.GetGeoTransform()  # crs transform information
    img_proj = img.GetProjection()  # projection
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)
    img_new = img_array.copy()
    for class_index in [1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        img_new[img_array == class_index] = 0
    img_new[img_new != 0] = 1
    writeTif(img_new[np.newaxis, :, :], os.path.join(root_dir, 'mask_new.tif'), True, img_geotrans,
             img_proj)

def match_extent(src_dir, dst_dir):
    src_img = gdal.Open(src_dir)
    dst_img = gdal.Open(dst_dir)
    geotrans = dst_img.GetGeoTransform()
    proj = dst_img.GetProjection()
    dst_array = dst_img.ReadAsArray()
    src_array = src_img.ReadAsArray()
    replace_img = np.zeros([1, dst_img.RasterYSize, dst_img.RasterXSize])
    # replace_img[0, 2:src_img.RasterYSize+2, 3:src_img.RasterXSize+3] = src_array
    replace_img[0,2:dst_img.RasterYSize-2,1:dst_img.RasterXSize-2] = src_array
    replace_img[replace_img == -128] = 0
    writeTif(replace_img, r'F:\DigitalAG\liheng\EU\theia\theia2019_matched.tif', require_proj=True, transform=geotrans, proj=proj)

if __name__ == '__main__':
    root_dir = r'F:\DigitalAG\liheng\EU\2019\raw data'
    save_dir = r'F:\DigitalAG\liheng\EU\2019\wt_sfl_cn\segmentation'
    patch_size = 257
    create_historical_CDL([os.path.join(r'F:\DigitalAG\liheng\EU\CDL', 'CDL_2016.tif'),
                           os.path.join(r'F:\DigitalAG\liheng\EU\CDL', 'CDL_2017.tif'),
                           os.path.join(r'F:\DigitalAG\liheng\EU\CDL', 'CDL_2018.tif'),
                           ], target_class=[1,2,3,4,5,6])
    reclassify_CDL(r'F:\DigitalAG\liheng\EU\RPG\rpg2016_matched.tif', year='2016')
    match_extent(r'F:\DigitalAG\liheng\EU\theia\theia2019.tif', r'F:\DigitalAG\liheng\EU\2019\raw data\CDL_2019.tif')
    reclassify_theia(os.path.join(root_dir, 'mask.tif'))
    for i in range(60, 61):
        try:
            img_path = os.path.join(root_dir, 'FR_2018_' + str(i) + '.tif')
            readImagePath(img_path, patch_size, patch_size, str(i), target=False, continue_count=True)
        except:
            continue

    target_path = os.path.join(root_dir, 'CDL_2018.tif')
    readImagePath(target_path, patch_size, patch_size, 'target', target=True, continue_count=True)

    target_path = os.path.join(root_dir, 'historical_CDL(wt_bt).tif')
    readImagePath(target_path, patch_size, patch_size, 'target_historical', target=True, continue_count=True, require_proj=True)

    target_path = os.path.join(root_dir, 'mask.tif')
    readImagePath(target_path, patch_size, patch_size, 'mask', target=True, continue_count=True)