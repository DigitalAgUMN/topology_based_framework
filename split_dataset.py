#!/usr/bin/env python
# coding: utf-8

# Split images into training, validation and testing sets

# In[5]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from skimage import io

# In[6]:


def split_data(dir, identifier='', year=2019, remove=True):
    training_path = os.path.join(dir, 'training_grid'+identifier+'.csv')
    validation_path = os.path.join(dir, 'validation_grid'+identifier+'.csv')
    testing_path = os.path.join(dir, 'testing_grid'+identifier+'.csv')
    for csv in [training_path, validation_path, testing_path]:
        if os.path.exists(csv) and remove:
            os.remove(csv)
    reference = io.imread(os.path.join(r'E:\DigitalAG\liheng\IW\sentinel-2\RDEG\2018\IW_CDL_2018.tif'))
    x_num = reference.shape[1] // 512
    y_num = reference.shape[0] // 512
    training = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    validation = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    testing = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    for date_index, date in enumerate(range(0, 24)):
        date_dir = os.path.join(dir, str(date))
        # target_dir = date_dir
        target_dir = os.path.join(dir, 'target')
        for image in os.listdir(date_dir):
            if image.endswith('tif'):
                img_index = int(image.split(".")[0])
                i = img_index // y_num
                j = img_index - (y_num * i)
                grid_x = i // (x_num // 3)
                grid_y = j // (y_num // 3)
                if grid_x == 3:
                    grid_x = 2
                if grid_y == 3:
                    grid_y = 2
                grid_index = grid_x * 3 + grid_y + 1

                if grid_index in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    training = training.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                                                                                  orient='index',columns=['img_dir', 'target_dir']), ignore_index=True)
                if grid_index in [9]:
                    validation = validation.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                                                                                  orient='index',columns=['img_dir', 'target_dir'] ), ignore_index=True)
                if grid_index in [9]:
                    testing = testing.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                                                                                  orient='index',columns=['img_dir', 'target_dir'] ), ignore_index=True)
    testing.to_csv(os.path.join(dir, 'testing_grid'+identifier+'.csv'), index=False)
    training.to_csv(os.path.join(dir, 'training_grid'+identifier+'.csv'), index=False)
    validation.to_csv(os.path.join(dir, 'validation_grid'+identifier+'.csv'), index=False)

def build_csv(dir, identifier='',remove=True):
    training_path = os.path.join(dir, 'training_grid'+identifier+'.csv')
    validation_path = os.path.join(dir, 'validation_grid'+identifier+'.csv')
    testing_path = os.path.join(dir, 'testing_grid'+identifier+'.csv')
    for csv in [training_path, validation_path, testing_path]:
        if os.path.exists(csv) and remove:
            os.remove(csv)
    training = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    validation = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    testing = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    for date_index, date in enumerate(range(0, 60)):
        try:
            date_dir = os.path.join(dir, str(date))
            # target_dir = date_dir
            target_dir = os.path.join(dir, 'target')
            for image in os.listdir(date_dir):
                if image.endswith('tif'):
                    img_index = int(image.split(".")[0])
                    training = training.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]}, orient='index',columns=['img_dir', 'target_dir']), ignore_index=True)
                    validation = validation.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]}, orient='index',columns=['img_dir', 'target_dir'] ), ignore_index=True)
                    testing = testing.append(pd.DataFrame.from_dict(data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]}, orient='index',columns=['img_dir', 'target_dir'] ), ignore_index=True)
        except:
            pass
    testing.to_csv(os.path.join(dir, 'testing_grid'+identifier+'.csv'), index=False)
    training.to_csv(os.path.join(dir, 'training_grid'+identifier+'.csv'), index=False)
    validation.to_csv(os.path.join(dir, 'validation_grid'+identifier+'.csv'), index=False)


if __name__ == '__main__':
    # codes for the first manuscript
    # for year in range(2018, 2019):
    #     root_dir = r'E:\DigitalAG\liheng\IW\sentinel-2\NIR\\' + str(year)
    #     split_data(root_dir, identifier='', year=year)
    build_csv(r'F:\DigitalAG\liheng\EU\2018\wt_bt\segmentation')




