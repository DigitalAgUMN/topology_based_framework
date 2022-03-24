#!/usr/bin/env python
# coding: utf-8
'''
This is the main function that run the topo
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from Dataset import BuildingDataset
import torch
import torch.utils as utils
from model import Unet
from trainer import Trainer
from torch.autograd import Variable
from utility_functions import project_to_target, save_fig, writeTif
import pandas as pd
from skimage import io

global training, prediction, cuda
training = True
prediction = False
cuda = False
seed = 11
crop = 'all'

root_dir = r'.\result'
model_dir = r'.\result\model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
train_dir = r'.\data\training\img'
vali_dir = r'.\data\training\img'
test_dir = r'.\data\testing\img'

hyperparameter = dict(
    batch_size=16,
    epochs=200,
    lr=0.001,  ## this param does not work for the cyclic LR
    optimizer='SGD',
    lr_scheduler='CLR',
    CLR_params=[0.01, 0.05, 25], ## applies only when using cyclic LR
    milestones = [10,20,30], ## applies only when using multistep LR
    gamma = 0.75,
    weight_decay=0,
    model='resnet',
    hidden_layer=[32, 64],
    weight=[1, 150, 300, 400],
    model_index='18',
    id='IA_corn_soy',
    crop=['corn', 'soybean']
)

def run(gpu=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('-------------------------')
        print(torch.backends.cudnn.version())
        print(torch.__version__)
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            print("There are {} CUDA devices".format(torch.cuda.device_count()))
            print("Setting torch GPU to {}".format(gpu))
            torch.cuda.set_device(gpu)
            print("Using device:{} ".format(torch.cuda.current_device()))
            torch.cuda.manual_seed(seed)

    # building the net
    model = Unet(features=hyperparameter['hidden_layer'])
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    if cuda:
        model = model.cuda()
    identifier = hyperparameter['id'] + '_' + hyperparameter['model_index']
    trainer = Trainer(net=model,
                      file_path=root_dir,
                      train_dir=train_dir,
                      vali_dir=vali_dir,
                      test_dir=test_dir,
                      model_dir=model_dir,
                      hyperparams=hyperparameter,
                      identifier=identifier,
                      cuda=cuda)
    # training
    if training:
        print("begin training!")
        trainer.train_model()

    if prediction:
        print("restore the model")
        for model_folder in os.listdir(model_dir):
            if model_folder.split('_')[-1] in ['notselected']:
                continue
            model_folder_path = os.path.join(model_dir, model_folder)
            if not os.path.isdir(model_folder_path):
                continue
            model = os.path.join(model_folder_path, str(model_folder) + '.pkl')
            accuracy_access(trainer, test_dir, model, model_folder)

def accuracy_access(trainer, test_dir, model, model_folder):
    trainer.restore_model(model, True)
    test_data = BuildingDataset(dir=test_dir, transform=None, target=False)
    test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
    precision = np.zeros([len(test_loader), 3])
    recall = np.zeros([len(test_loader), 3])
    count = 0
    for l, sample in enumerate(test_loader):
        image = Variable(sample['image'], requires_grad=False)
        patch = sample['patch']
        label = io.imread(os.path.join(test_dir.replace('_img', '_target'), patch[0] + '.tif'))
        if cuda:
            image = image.cuda()
        pred = trainer.predict(image)
        filter = torch.ge(torch.max(torch.nn.functional.softmax(pred, dim=1), dim=1)[0], 0.995).type(
            torch.cuda.FloatTensor)
        pred = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1)
        cm, pred_img, target_img = project_to_target(pred, patch, 257, id=id,
                                                     classes=[0, 1, 2, 3],
                                                     use_cm=True, use_mask=False)
            ## to convert soybean index
            # pred_img[pred_img == 2] = 3

        if (pred_img != 0).sum() > 257 ** 2 * 0.01:
            recall[l, 0] = cm[1, 1] / cm[1, :].sum()
            precision[l, 0] = cm[1, 1] / cm[:, 1].sum()
            recall[l, 1] = cm[2, 2] / cm[2, :].sum()
            precision[l, 1] = cm[2, 2] / cm[:, 2].sum()
            recall[l, 2] = cm[3, 3] / cm[3, :].sum()
            precision[l, 2] = cm[3, 3] / cm[:, 3].sum()
            count += 1
            result_folder = os.path.join(root_dir, 'result')
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            writeTif(pred_img, os.path.join(result_folder, str(patch[0]) + '.tif'))
            save_fig(pred, label, image, pred_img, target_img, r'F:\DigitalAG\liheng\EU\\'+id+r'\sample',
                      patch[0], 'all', title = 'pre:{}, rec:{}'.format(np.round(precision[l], 4), np.round(recall[l], 4)))
    precision_wheat = precision[np.logical_and(precision[:, 0] != 0, np.isnan(precision[:, 0]) == False), 0].mean()
    precision_corn = precision[np.logical_and(precision[:, 1] != 0, np.isnan(precision[:, 1]) == False), 1].mean()
    precision_sunflower = precision[np.logical_and(precision[:, 2] != 0, np.isnan(precision[:, 2]) == False), 2].mean()

    # print(
    #     'the corn precision is {}, the soybean recall is {},  the count is {}'.format(
    #         precision_corn, precision_soy, count))
    if not os.path.exists(os.path.join(model_dir, 'accuracy2.csv')):
        df = pd.DataFrame(data=[], columns=['model_name', 'precision_wheat', 'precision_corn', 'precision_sunflower', 'count', 'info'])
        df.to_csv(os.path.join(model_dir, 'accuracy2.csv'), index=False)
    df = pd.read_csv(os.path.join(model_dir, 'accuracy2.csv'))
    df = df.append(pd.DataFrame.from_dict(
        data={0: [model_folder, precision_wheat, precision_corn, precision_sunflower, count, 'corn/soybean']},
        orient='index', columns=['model_name', 'precision_wheat', 'precision_corn', 'precision_sunflower', 'count', 'info']), ignore_index=True)
    df.to_csv(os.path.join(model_dir, 'accuracy2.csv'), index=False)

if __name__ == "__main__":
    run()
