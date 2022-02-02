#!/usr/bin/env python
# coding: utf-8

# In[70]:


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from dataio import BuildingDataset
import torch
import torch.utils as utils
from model import Unet, Unet_3
import wandb
from trainer import Trainer
from torch.autograd import Variable
from utility_functions import project_to_target, save_fig, writeTif, save_fig_spectral
import pandas as pd
from dataio import check_data_target
from skimage import io
from dataio import scale_percentile_n


global training, prediction, cuda
training = True
prediction = False
cuda = True  # 是否使用GPU
seed = 11
mode = 'multiple'
crop = 'all'
id = 'wt_sfl_cn'

root_dir = r'F:\DigitalAG\liheng\EU\\' + id
model_dir = r'F:\DigitalAG\liheng\EU\\' + id + r'\model'

train_dir = os.path.join(root_dir, r'training_img')
vali_dir = os.path.join(root_dir, r'training_img')
test_dir = os.path.join(root_dir, r'training_img')

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
    id=id,
    crop = ['winter wheat', 'corn', 'sunflowers']
)


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
            # label = label.cuda()
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

def run(gpu=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('-------------------------')
        print(torch.backends.cudnn.version())
        print(torch.__version__)
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            # os.environ['CUDA_ENABLE_DEVICES'] = '0'
            print("There are {} CUDA devices".format(torch.cuda.device_count()))
            print("Setting torch GPU to {}".format(gpu))
            torch.cuda.set_device(gpu)
            print("Using device:{} ".format(torch.cuda.current_device()))
            torch.cuda.manual_seed(seed)
            # torch.backends.cudnn.enabled = False

    # building the net
    model = Unet_3(features=hyperparameter['hidden_layer'])
    # model = ResNetUNet(1)
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    if cuda:
        model = model.cuda()
    identifier = hyperparameter['id'] + '_' + hyperparameter['model_index']
    trainer = Trainer(net=model, file_path=root_dir, train_dir=train_dir, vali_dir=vali_dir, test_dir=test_dir,
                      model_dir=model_dir,
                      hyperparams=hyperparameter, identifier=identifier, cuda=cuda)

    # training
    if training:
        print("begin training!")
        wandb.init(entity='chenxilin', project='france', name=identifier,
                   config=hyperparameter)
        trainer.train_model()

    if prediction:
        print("restore the model")
        for model_folder in os.listdir(model_dir):
            if model_folder.split('_')[-1] in ['notselected']:
                continue
            model_folder_path = os.path.join(model_dir, model_folder)
            if not os.path.isdir(model_folder_path):
                continue
            if mode == 'single':
                model = os.path.join(model_folder_path, str(model_folder) + '.pkl')
                accuracy_access(trainer, test_dir, model, model_folder)
            if mode == 'multiple':
                for epoch in range(173,174):
                    current_epoch = model_folder.split('_')[-1]
                    model = os.path.join(model_folder_path, model_folder.replace(current_epoch, str(epoch)) + '.pkl')
                    if model_folder.replace(current_epoch, str(epoch)) + '.pkl' not in os.listdir(model_folder_path):
                        continue
                    accuracy_access(trainer, test_dir, model, model_folder.replace(current_epoch, str(epoch)) + '.pkl')

if __name__ == "__main__":
    run()
