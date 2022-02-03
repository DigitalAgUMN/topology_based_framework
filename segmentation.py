#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from osgeo import gdal
import os
from skimage import io

def non_overlap_segmentation(img,
                             folder,
                             x_range,
                             y_range,
                             target=False,
                             require_proj=False,
                             continue_count=False):
    """
    Segment image tile by tile
    x_range: int, the width of image patches
    y_range: int, the height of image patches
    target: bool, determine whether the input image is satellite image or ground truth, e.g., CDL
    require_proj: bool, whether to add georeference to the output image
    continue_count: bool, whether to use coutinuous index for image patches in different time step
    """
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
                valid_ratio = (patch != 0).mean()
                if valid_ratio < 0.8: ## determine if the patch has enough valid pixel
                    continue
            if target:
                patch = img_array[y_off_patch:y_off_patch + y_range,
                                  x_off_patch:x_off_patch + x_range][np.newaxis, :, :]
            patch_name = os.path.join(folder, str(i * y_num + j + original_count) + '.tif')
            if require_proj:
                new_top_left_x = top_left_x + x_off_patch * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - y_off_patch * np.abs(n_s_pixel_resolution)
                dst_transform = (
                    new_top_left_x,
                    img_geotrans[1],
                    img_geotrans[2],
                    new_top_left_y,
                    img_geotrans[4],
                    img_geotrans[5])
                writeTif(patch, patch_name,
                         require_proj=True,
                         transform=dst_transform,
                         proj=img_proj)
            else:
                writeTif(patch, patch_name)

def random_segmentation(img,
                        folder,
                        x_range,
                        y_range,
                        target=False,
                        require_proj=False,
                        mask=None):
    """
    Segment image by randomly choosing tiles
    img: ndarray, satellite image to be segmented
    x_range: int, the width of image patches
    y_range: int, the height of image patches
    target: bool, determine whether the input image is satellite image or ground truth, e.g., CDL
    require_proj: bool, whether to add georeference to the output image
    mask: optional, ndarray, a mask to help filter out randomly selected patches
    """
    if require_proj:
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        top_left_x = img_geotrans[0]
        w_e_pixel_resolution = img_geotrans[1]
        top_left_y = img_geotrans[3]
        n_s_pixel_resolution = img_geotrans[5]
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)
    n = 10000 # the number of patches
    radius = (x_range - 1) // 2
    x_coor = np.random.choice(range(radius, x_size - radius - 1), n)
    y_coor = np.random.choice(range(radius, y_size - radius - 1), n)
    for i in range(0, n):
        if not target:
            patch = img_array[:, y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
            valid_ratio = (patch != 0).mean()
            if valid_ratio < 0.8: ## determine if the patch has enough valid pixel
                continue
            ### an example of using mask to filter image patches ###
            '''
            only when corn (code 1) and soybean (code 5) are both larger than 20% will the patch be kept
            '''
            if mask:
                mask_patch = mask[y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
                if (mask_patch == 1).sum() < 0.2 * x_range * y_range and \
                        (mask_patch == 5).sum() < 0.2 * x_range * y_range:
                    continue
            ### an example of using mask to filter image patches ###
            patch_name = os.path.join(folder, str(x_coor[i]) + '_' + str(y_coor[i]) + '.tif')
            if require_proj:
                new_top_left_x = top_left_x + (x_coor[i] - 128) * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - (y_coor[i] - 128) * np.abs(n_s_pixel_resolution)
                dst_transform = (
                    new_top_left_x,
                    img_geotrans[1],
                    img_geotrans[2],
                    new_top_left_y,
                    img_geotrans[4],
                    img_geotrans[5])
                writeTif(patch, patch_name,
                         require_proj=True,
                         transform=dst_transform,
                         proj=img_proj)
            else:
                writeTif(patch, patch_name)
        if target:
            patch = img_array[y_coor[i] - radius:y_coor[i] + radius + 1,
                    x_coor[i] - radius:x_coor[i] + radius + 1][np.newaxis, :, :]
            if mask:
                mask_patch = mask[y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
                if (mask_patch == 1).sum() < 0.2 * x_range * y_range and \
                        (mask_patch == 5).sum() < 0.2 * x_range * y_range:
                    continue
                patch_name = os.path.join(folder, str(x_coor[i]) + '_' + str(y_coor[i]) + '.tif')
                if require_proj:
                    new_top_left_x = top_left_x + (x_coor[i] - 128) * np.abs(w_e_pixel_resolution)
                    new_top_left_y = top_left_y - (y_coor[i] - 128) * np.abs(n_s_pixel_resolution)
                    dst_transform = (
                        new_top_left_x,
                        img_geotrans[1],
                        img_geotrans[2],
                        new_top_left_y,
                        img_geotrans[4],
                        img_geotrans[5])
                    writeTif(patch, patch_name,
                             require_proj=True,
                             transform=dst_transform,
                             proj=img_proj)
                else:
                    writeTif(patch, patch_name)

def writeTif(bands,
             path,
             require_proj=False,
             transform=None,
             proj=None):
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

def readImagePath(img_path,
                  x_range,
                  y_range,
                  folder_name='',
                  segmentation_approach='nonoverlap',
                  target=False,
                  require_proj=True,
                  continue_count=True):
    img = gdal.Open(img_path)
    folder = os.path.join(save_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(folder + ' has been created')
    if not continue_count:
        [os.remove(os.path.join(folder, file)) for file in os.listdir(folder)]
    if segmentation_approach == 'nonoverlap':
        non_overlap_segmentation(img, folder, x_range, y_range,
                                 target=target,
                                 require_proj=require_proj)
    if segmentation_approach == 'random':
        random_segmentation(img, folder, x_range, y_range,
                            target=target,
                            require_proj=require_proj,
                            mask=io.imread(os.path.join(root_dir, 'CDL_2018.tif')))

def create_historical_CDL(img_list, target_class=[0]):
    target = []
    for t in img_list:
        img = gdal.Open(t)
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
    writeTif(historical_CDL[np.newaxis,:,:], os.path.join(root_dir, 'historical_CDL.tif'), True, img_geotrans, img_proj)

np.random.seed(122)
if __name__ == '__main__':
    root_dir = r'.\data\raw data'
    save_dir = r'.\data\segmentation'
    patch_size = 512
    create_historical_CDL([os.path.join(root_dir, 'CDL_2017.tif'),
                           os.path.join(root_dir, 'CDL_2018.tif')],
                          target_class=[1,5])

    for i in range(0, 24):
        img_path = os.path.join(root_dir, '2018_' + str(i) + '.tif')
        readImagePath(img_path, patch_size, patch_size, str(i), target=False, continue_count=True)


    target_path = os.path.join(root_dir, 'CDL_2018.tif')
    readImagePath(target_path, patch_size, patch_size, 'target', target=True, continue_count=True)

    target_path = os.path.join(root_dir, 'historical_CDL.tif')
    readImagePath(target_path, patch_size, patch_size, 'target_historical',
                  target=True,
                  continue_count=True,
                  require_proj=True)
