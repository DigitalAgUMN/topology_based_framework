#!/usr/bin/env python
# coding: utf-8
"""
@author: Chenxi
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utilities import get_hist2d, get_target, get_separability
from utilities import get_coordinate

np.random.seed(2000)
class generate_heat_map(object):
    def __init__(self, input_dir, out_dir, type='typeI', bins=129, scales=[0.3, 0.6]):
        '''
        input_dir: str, the directory containing the training/validation/testing csv files
        out_dir: str, the directory to store the output heat maps
        type: str, how to generate heat maps
              -- typeI means to generate typeI heat maps (target image indicates the locations of crops)
              -- typeII means to generate typeII heat maps (target image is blank)
              -- testing means to generate heat maps for testing.
                 No need to distinguish typeI and typeII in this case.
                 No target image will be generated for each heat map
        scales: list, the scale used to generate heat maps, see the paper for more details
        bins, int, the bin size used to generate heat maps, see the paper for more details
        '''
        assert type in ['typeI', 'typeII', 'testing'], \
            'type can only be one of "typeI", "typeII" and "testing"'
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.type = type
        self.scales = scales
        self.bins = bins
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def make_folders(self, folder_id):
        '''
        create folders to store heat maps (.tiff), the target of heat maps (.tiff)
        '''
        if self.type == 'testing':
            folders = ['_input']
        else:
            folders = ['_input', '_target']
        for item in folders:
            if not os.path.exists(os.path.join(self.out_dir, folder_id + item)):
                os.makedirs(os.path.join(self.out_dir, folder_id + item))

    def isValidImg(self, arr, threshold):
        return True if (arr != 0).mean() < threshold else False

    def reclassify(self, crop_index, label):
        label_recls = np.zeros_like(label)
        for i, index in enumerate(crop_index):
            label_recls[label == index] = i + 1
        return label_recls

    def heat_map(self, dir):
        '''
        dir: str, the file containing the directories of input satellite images and corresponding labels
        generate heat maps and their targets from labelled satellite imagery
        '''
        self.make_folders(self.type)
        img_label_dirs = np.array(pd.read_csv(dir)[['img_dir', 'label_dir']])
        np.random.shuffle(img_label_dirs)
        for index, img_label in enumerate(img_label_dirs):
            ###### generate heat maps of satellite image patches ######
            ###########################################################
            img = io.imread(img_label[0]) / 10000
            label = io.imread(img_label[1])
            label_historical = io.imread(img_label[1].replace('label', 'label_historical'))
            '''
            create a 4D array (batch-channel-height-width)
            '''
            img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
            b, c, h, w = img.shape
            '''
            Discard the patch if:
            -- more than 20% pixels are nodata, 
            -- more than 50% pixels are not planted with target crops in historical years
            '''
            if self.isValidImg(img, 0.8) or (label_historical == 0).sum() > h * w * 0.5:
                continue
            '''
            -- In this example, we are classifying corn and soybeans, therefore, we used index 1 and 5
            -- Users can change the index based on their own label
            '''
            label_recls = self.reclassify([1, 5], label)
            hm, _, _ = get_hist2d(img, bins=self.bins, scales=self.scales)
            hm_label, _, _ = get_hist2d(img, bins=self.bins, label=label_recls, scales=self.scales) ## the heat map with label information, will be used for generate targets
            hm_historical, _, _ = get_hist2d(img, bins=self.bins, label=label_historical, scales=self.scales)
            ###### generate heat maps of satellite image patches ######
            ###########################################################

            ###### generate targets for heat maps ######
            ############################################
            sep = get_separability(img, label_recls)  ## JM distance between crops
            if self.type == 'typeI':
                criterion = (sep > 0.9).sum() != len(sep.flatten())
            if self.type == 'typeII':
                criterion = (sep > 0.9).sum() == len(sep.flatten())
            if self.type == 'testing':
                criterion = False
            if criterion:
                continue
            x_coor = get_coordinate(hm)
            y_coor = get_coordinate(hm, x_coor=False)
            data_target = get_target(
                hm_label,
                separability=sep,
                percentile_pure=50,
                threshold_separability=0.9,
                crop='all'
                )
            ###### generate targets for heat maps ######
            ############################################
            ###### save heat maps and targets ######
            ########################################
            '''
            Note that the variable 'img_name' and 'date' is retrieved based on the user-defined naming convension
            users may change the following two lines of code based on their own convention
            '''
            img_name = img_label[0].split('.')[1].split('\\')[-1]
            date = int(img_label[0].split('.')[1].split('\\')[-2])
            name = str(date) + '_' + str(img_name) + '.tif'
            io.imsave(os.path.join(os.path.join(self.out_dir, self.type + '_input'), name),
                      np.concatenate((hm, hm_historical[:, 0:1, :, :],
                                      x_coor[np.newaxis, np.newaxis, :, :],
                                      y_coor[np.newaxis, np.newaxis, :, :]), axis=1).squeeze().astype(
                          np.int16))
            if self.type != 'testing':
                io.imsave(os.path.join(os.path.join(self.out_dir, self.type + '_target'), name),
                          data_target.squeeze().astype(np.int8))
            ###### save heat maps and targets ######
            ########################################

    def visualize(self):
        from matplotlib.colors import ListedColormap
        '''
        -- xticks and yticks are based the bin size 128
        -- users might change the value based on their own bin sizes
        '''
        xticks = {'raw': [0, 31, 63, 95, 127],
                  'scale': [round(i, 1) for i in np.linspace(0, self.scales[0], 5)]}
        yticks = {'raw': [0, 31, 63, 95, 127],
                  'scale': [round(i, 1) for i in np.linspace(self.scales[1], 0, 5)]}
        img_dir = os.path.join(self.out_dir, self.type+'_input')
        for dir in os.listdir(img_dir):
            img = io.imread(os.path.join(img_dir, dir))
            target = io.imread(os.path.join(img_dir.replace('input', 'target'), dir))
            fig = plt.figure(1, figsize=(14, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, :, 0], cmap='gist_ncar_r')
            plt.xticks(xticks['raw'], xticks['scale'])
            plt.yticks(yticks['raw'], yticks['scale'])
            plt.subplot(1, 3, 2)
            plt.imshow(img[:, :, 1], cmap='gist_ncar_r')
            plt.xticks(xticks['raw'], xticks['scale'])
            plt.yticks(yticks['raw'], yticks['scale'])
            plt.subplot(1, 3, 3)
            color_list = np.array([[255, 255, 255], [255, 211, 0], [38, 112, 0]]) / 255.0
            cmap = ListedColormap(color_list[:target.shape[2]])
            plt.imshow(target.sum(axis=2), cmap=cmap)
            plt.xticks(xticks['raw'], xticks['scale'])
            plt.yticks(yticks['raw'], yticks['scale'])
            folder_name = self.type + '_visualize'
            if not os.path.exists(os.path.join(self.out_dir, folder_name)):
                os.makedirs(os.path.join(self.out_dir, folder_name))
            plt.savefig(os.path.join(os.path.join(self.out_dir, folder_name), dir))
            plt.cla()

if __name__ == '__main__':

    ## an example to generate typeI heat maps
    typeI_generator = generate_heat_map(r'.\data\segmentation', r'.\result\heatmaps', type='typeII', bins=129, scales=[0.3, 0.6])
    typeI_generator.heat_map(os.path.join(typeI_generator.input_dir, 'training.csv'))
    typeI_generator.visualize()

    # generator = generate_heat_map('please enter your own input path',
    #                                       'please enter your own output path',
    #                                       type='one of typeI, typeII or testing',
    #                                       bines='xxx', scales='[xxx, xxx]')
    # generator.heat_map(os.path.join(generator.input_dir, 'please enter your own file path'))
    # generator.visualize()


