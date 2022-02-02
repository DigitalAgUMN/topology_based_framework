#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from gdalconst import *
from utility_functions import get_patch, get_hist2d, plot_hist2d, \
                      get_separability, get_valid_ratio, get_target, \
                      get_prob
from gee_functions import get_box_grid
import ee
import geemap

# ee.Initialize()
# # aoi = ee.Geometry.Polygon(
# #   [[
# #     [-94.88635811097191,44.42734494862467],
# #     [-94.23267158753441,44.42734494862467],
# #     [-94.23267158753441,44.84948055853524],
# #     [-94.88635811097191,44.84948055853524],
# #     [-94.88635811097191,44.42734494862467]
# #    ]], None, False)
# def decodeQA60(img):
#     qa60 = img.select('QA60').updateMask(img.select('B2'))
#     cloudBitMask = qa60.bitwiseAnd(ee.Number(2).pow(10).int())
#     cirrusBitMask = qa60.bitwiseAnd(ee.Number(2).pow(11).int())
#     clear = cloudBitMask.eq(0).And(cirrusBitMask.eq(0)).rename(['PXQA60_CLEAR']).toInt()
#     clear = clear.updateMask(clear)
#     return img.addBands([clear])
#
# def applyCloudmask(img):
#   clearmask = img.select('PXQA60_CLEAR')
#   return img.updateMask(clearmask)
#
# data = ee.ImageCollection("COPERNICUS/S2_SR").filterDate('2019-05-01', '2019-05-06').filterBounds(aoi).map(decodeQA60).map(applyCloudmask)
# grid = get_box_grid(ee.Feature(aoi), 10, 10)
#
# def decodeQA60(img):
#     qa60 = img.select('QA60').updateMask(img.select('B2'))
#     cloudBitMask = qa60.bitwiseAnd(ee.Number(2).pow(10).int())
#     cirrusBitMask = qa60.bitwiseAnd(ee.Number(2).pow(11).int())
#     clear = cloudBitMask.eq(0).And(cirrusBitMask.eq(0)).rename(['PXQA60_CLEAR']).toInt()
#     clear = clear.updateMask(clear)
#     return img.addBands([clear])
#
# def applyCloudmask(img):
#   clearmask = img.select('PXQA60_CLEAR')
#   return img.updateMask(clearmask)
#
# data = ee.ImageCollection("COPERNICUS/S2_SR").filterDate('2019-05-01', '2019-05-06').filterBounds(aoi).map(decodeQA60).map(applyCloudmask)
# def clip_grid(grid):
#     return data.mosaic().select(['B5']).clip(grid)
# test = grid.map(clip_grid)
# Map = geemap.Map()
# print (Map.addLayer(test))

require_proj = False
CDL_path = 'D:\My Drive\Digital_Agriculture\Liheng\CDL.tif'
RDED_path = 'D:\My Drive\Digital_Agriculture\Liheng\RDED.tif'
SWIR_path = 'D:\My Drive\Digital_Agriculture\Liheng\SWIR.tif'
CDL = gdal.Open(CDL_path, GA_ReadOnly)
RDED = gdal.Open(RDED_path, GA_ReadOnly)
SWIR = gdal.Open(SWIR_path, GA_ReadOnly)

if require_proj:
    geotrans = CDL.GetGeoTransform()  # 获取仿射矩阵信息
    proj = CDL.GetProjection()  # 获取投影信息
    top_left_x = geotrans[0]  # 左上角x坐标
    w_e_pixel_resolution = geotrans[1]  # 东西方向像素分辨率
    top_left_y = geotrans[3]  # 左上角y坐标
    n_s_pixel_resolution = geotrans[5]  # 南北方向像素分辨率

CDL_array = CDL.ReadAsArray(0, 0, CDL.RasterXSize, CDL.RasterYSize).astype(np.int)
RDED_array = RDED.ReadAsArray(0, 0, RDED.RasterXSize, RDED.RasterYSize).astype(np.int)[:, np.newaxis, :, :]
SWIR_array = SWIR.ReadAsArray(0, 0, SWIR.RasterXSize, SWIR.RasterYSize).astype(np.int)[:, np.newaxis, :, :]
arr = np.concatenate((RDED_array, SWIR_array), axis=1)[6:12]
CDL_patch = get_patch(512, CDL_array)
S2_patch = get_patch(512, arr)

'''
choose dates with valid pixels larger than 80% of the total
'''
ratio_valid = get_valid_ratio(S2_patch)
indi_select = ratio_valid > 0.8
S2_patch = S2_patch[indi_select]

"""
convert the image to 3 classes:
0 for background
1 for corn
2 for soybean
"""
code_class = [1, 5]
arr_class = np.zeros_like(CDL_patch)
for i_cur, code_cur in enumerate(code_class):
    arr_class[CDL_patch==code_cur] = i_cur + 1

list_img, list_xedges, list_yedges = get_hist2d(S2_patch, arr_class=arr_class)
list_img, list_xedges, list_yedges = get_hist2d(S2_patch)
plot_hist2d(list_img, list_xedges, list_yedges)

separability = get_separability(S2_patch, CDL_patch)
data_target = get_target(
    list_img,
    separability=separability, threshold_separability=0.1
)
plot_hist2d(np.clip(data_target, 0, 1), list_xedges, list_yedges)