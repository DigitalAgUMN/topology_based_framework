#!/usr/bin/env python
# coding: utf-8

import numpy as np
from osgeo import gdal
import os
from skimage import io
import pandas as pd

def non_overlap_segmentation(img,
                             folder,
                             x_range,
                             y_range,
                             target=False,
                             require_proj=False):
    """
    Segment image tile by tile
    x_range: int, the width of image patches
    y_range: int, the height of image patches
    target: bool, determine whether the input image is satellite image or ground truth, e.g., CDL
    require_proj: bool, whether to add georeference to the output image
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
    original_count = 0
    for i in range(0, x_num):
        for j in range(0, y_num):
            x_off_patch = i * x_range
            y_off_patch = j * y_range
            if not target:
                patch = img_array[:, y_off_patch:y_off_patch + y_range,
                        x_off_patch:x_off_patch + x_range]
                if (patch != 0).mean() < 0.8: ## determine if the patch has enough valid pixel
                    continue
            if target:
                patch = img_array[y_off_patch:y_off_patch + y_range,
                                  x_off_patch:x_off_patch + x_range][np.newaxis, :, :]
            patch_name = os.path.join(folder, str(i * y_num + j + original_count) + '.tif')
            if require_proj:
                new_top_left_x = top_left_x + x_off_patch * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - y_off_patch * np.abs(n_s_pixel_resolution)
                dst_transform = (new_top_left_x, img_geotrans[1], img_geotrans[2],
                                 new_top_left_y, img_geotrans[4], img_geotrans[5])
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
                        number_of_patch=1000,
                        target=False,
                        require_proj=False,
                        mask=None):
    """
    Segment image by randomly choosing tiles
    img: ndarray, satellite image to be segmented
    x_range: int, the width of image patches
    y_range: int, the height of image patches
    number_of_patch: int, the number of patches to randomly generate
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
    radius = (x_range - 1) // 2
    x_coor = np.random.choice(range(radius, x_size - radius - 1), number_of_patch)
    y_coor = np.random.choice(range(radius, y_size - radius - 1), number_of_patch)
    for i in range(0, number_of_patch):
        if not target:
            patch = img_array[:, y_coor[i] - radius:y_coor[i] + radius + 1,
                    x_coor[i] - radius:x_coor[i] + radius + 1]
            if (patch != 0).mean() < 0.8: ## determine if the patch has enough valid pixel
                continue

            ########################################################
            ### an example of using mask to filter image patches ###
            ########################################################
            '''
            only when corn (code 1) and soybean (code 5) are both larger 
            than 20% will the patch be kept
            '''
            if mask:
                mask_patch = mask[y_coor[i] - radius:y_coor[i] + radius + 1, x_coor[i] - radius:x_coor[i] + radius + 1]
                if (mask_patch == 1).sum() < 0.2 * x_range * y_range and \
                        (mask_patch == 5).sum() < 0.2 * x_range * y_range:
                    continue
            ########################################################
            ### an example of using mask to filter image patches ###
            ########################################################

            patch_name = os.path.join(folder, str(x_coor[i]) + '_' + str(y_coor[i]) + '.tif')
            if require_proj:
                new_top_left_x = top_left_x + (x_coor[i] - radius) * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - (y_coor[i] - radius) * np.abs(n_s_pixel_resolution)
                dst_transform = (
                    new_top_left_x, img_geotrans[1], img_geotrans[2],
                    new_top_left_y, img_geotrans[4], img_geotrans[5])
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
                    new_top_left_x = top_left_x + (x_coor[i] - radius) * np.abs(w_e_pixel_resolution)
                    new_top_left_y = top_left_y - (y_coor[i] - radius) * np.abs(n_s_pixel_resolution)
                    dst_transform = (
                        new_top_left_x, img_geotrans[1], img_geotrans[2],
                        new_top_left_y, img_geotrans[4], img_geotrans[5])
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
                  require_proj=True):

    img = gdal.Open(img_path)
    folder = os.path.join(save_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(folder + ' has been created')
    [os.remove(os.path.join(folder, file)) for file in os.listdir(folder)]
    if segmentation_approach == 'nonoverlap':
        non_overlap_segmentation(img, folder, x_range, y_range,
                                 target=target,
                                 require_proj=require_proj)
    if segmentation_approach == 'random':
        random_segmentation(img, folder, x_range, y_range,
                            number_of_patch=2000,
                            target=target,
                            require_proj=require_proj,
                            mask=io.imread(os.path.join(root_dir, 'CDL_2018.tif')))

def generate_historical_map(img_list,
                            target_class=[0]):
    '''
    generate historical crops maps based on user-defined criterion
    img_list: list, contains directories of all historical data
    target class: list, contains crop class codes based on which the historical map will be created
                        for example, in CDL, 1 denotes corn and 5 denotes soybeans, target_class=[1,5]
                        indicates that all pixels that were not planted with corn or soybeans will be
                        marked as 0, otherwise, 1.
    '''
    target = []
    for img in img_list:
        img = gdal.Open(img)
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
        img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)
        target.append(img_array)
    historical_map = np.ones([y_size, x_size])
    for i in range(0, len(img_list)):
        for c in target_class:
            historical_map *= (target[i] != c)
    historical_map = -historical_map + 1
    writeTif(historical_map[np.newaxis, :, :],
             os.path.join(root_dir, 'historical_map.tif'),
             require_proj=True,
             transform=img_geotrans,
             proj=img_proj)

def build_csv(dir,
              identifier='',
              mode='training',
              remove=True):
    '''
    store information of training, validation or testing dataset into csv files
    dir: str, the directory storing image patches
    identifier: str, an unique identifier to name the csv file
    mode: str, if 'training', will build training and validation dataset from training years
               if 'testing', will build testing dataset from testing years
               e.g., if you want to train your model using 2017 ana 2018 data and test your model in 2019,
               then choose 'training' for 2017 and 2018 and 'testing' for 2019
    remove: bool, whether to delete existing csv files or not
    '''
    training_path = os.path.join(dir, 'training'+identifier+'.csv')
    validation_path = os.path.join(dir, 'validation'+identifier+'.csv')
    testing_path = os.path.join(dir, 'testing'+identifier+'.csv')
    for csv in [training_path, validation_path, testing_path]:
        if os.path.exists(csv) and remove:
            os.remove(csv)
    training = pd.DataFrame(data=[], columns=['img_dir', 'label_dir'])
    validation = pd.DataFrame(data=[], columns=['img_dir', 'label_dir'])
    testing = pd.DataFrame(data=[], columns=['img_dir', 'label_dir'])
    for date_index, date in enumerate(range(0, time_steps)):
        date_dir = os.path.join(dir, str(date))
        target_dir = os.path.join(dir, 'label')
        for image in os.listdir(date_dir):
            if image.endswith('tif'):
                img_index = int(image.split(".")[0])
                if mode == 'training':
                    ## split all patches into training and validation at a ratio of 7:3
                    indicator = np.random.choice(2, 1, p=[0.7, 0.3])
                    if indicator == 0:
                        training = training.append(pd.DataFrame.from_dict(
                            data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                            orient='index',
                            columns=['img_dir', 'label_dir']),
                            ignore_index=True)
                    if indicator == 1:
                        validation = validation.append(pd.DataFrame.from_dict(
                            data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                            orient='index',
                            columns=['img_dir', 'label_dir']),
                            ignore_index=True)
                if mode == 'testing':
                    testing = testing.append(pd.DataFrame.from_dict(
                        data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                        orient='index',
                        columns=['img_dir', 'label_dir']),
                        ignore_index=True)
    if mode == 'training':
        training.to_csv(os.path.join(dir, 'training'+identifier+'.csv'), index=False)
        validation.to_csv(os.path.join(dir, 'validation'+identifier+'.csv'), index=False)
    if mode == 'testing':
        testing.to_csv(os.path.join(dir, 'testing'+identifier+'.csv'), index=False)

np.random.seed(1000)
if __name__ == '__main__':
    root_dir = r'.\data\raw data'
    save_dir = r'.\data\segmentation'
    patch_size = 256
    time_steps = 24
    ## In this example, we only used a single-year CDL to create the historical map
    ## Users can use a longer time period when more historical data is available
    generate_historical_map([os.path.join(root_dir, 'CDL_2017.tif')],
                           target_class=[1, 5])
    ## Segmentation for satellite imagery in different time steps
    for i in range(0, time_steps):
        img_path = os.path.join(root_dir, '2018_' + str(i) + '.tif')
        readImagePath(img_path,
                      patch_size,
                      patch_size,
                      folder_name=str(i))
    label_path = os.path.join(root_dir, 'CDL_2018.tif')
    readImagePath(label_path,
                  patch_size,
                  patch_size,
                  folder_name='label',
                  target=True)
    historical_label_path = os.path.join(root_dir, 'historical_map.tif')
    readImagePath(historical_label_path,
                  patch_size,
                  patch_size,
                  folder_name='label_historical',
                  target=True)
    build_csv(r'.\data\segmentation')
