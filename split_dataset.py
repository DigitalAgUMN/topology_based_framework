#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from skimage import io

def build_csv(dir, identifier='', remove=True):
    training_path = os.path.join(dir, 'training_grid'+identifier+'.csv')
    validation_path = os.path.join(dir, 'validation_grid'+identifier+'.csv')
    testing_path = os.path.join(dir, 'testing_grid'+identifier+'.csv')
    for csv in [training_path, validation_path, testing_path]:
        if os.path.exists(csv) and remove:
            os.remove(csv)
    training = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    validation = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    testing = pd.DataFrame(data=[], columns=['img_dir', 'target_dir'])
    for date_index, date in enumerate(range(0, 24)):
        date_dir = os.path.join(dir, str(date))
        target_dir = os.path.join(dir, 'target')
        for image in os.listdir(date_dir):
            if image.endswith('tif'):
                img_index = int(image.split(".")[0])
                training = training.append(pd.DataFrame.from_dict(
                    data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                    orient='index',
                    columns=['img_dir', 'target_dir']),
                    ignore_index=True)
                validation = validation.append(pd.DataFrame.from_dict(
                    data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                    orient='index',
                    columns=['img_dir', 'target_dir']),
                    ignore_index=True)
                testing = testing.append(pd.DataFrame.from_dict(
                    data={img_index: [os.path.join(date_dir, image), os.path.join(target_dir, image)]},
                    orient='index',
                    columns=['img_dir', 'target_dir']),
                    ignore_index=True)
    testing.to_csv(os.path.join(dir, 'testing_grid'+identifier+'.csv'), index=False)
    training.to_csv(os.path.join(dir, 'training_grid'+identifier+'.csv'), index=False)
    validation.to_csv(os.path.join(dir, 'validation_grid'+identifier+'.csv'), index=False)

if __name__ == '__main__':
    build_csv(r'.\data\segmentation')




