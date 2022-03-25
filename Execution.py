#!/usr/bin/env python
# coding: utf-8

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from Model import Unet
from Trainer import Trainer

class Execution(object):
    def __init__(self, train_dir, vali_dir, test_dir, result_dir, hyperparameter, cuda=False):
        self.train_dir = train_dir
        self.vali_dir = vali_dir
        self.test_dir = test_dir
        self.result_dir = result_dir
        self.hyperparameter = hyperparameter
        self.cuda = cuda
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    def train(self):
        if torch.cuda.is_available():
            print('-------------------------')
            print(torch.backends.cudnn.version())
            print(torch.__version__)
            if not self.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                print("There are {} CUDA devices".format(torch.cuda.device_count()))
                print("Setting torch GPU to {}".format(0))
                torch.cuda.set_device(0)
                print("Using device:{} ".format(torch.cuda.current_device()))
                torch.cuda.manual_seed(0)
        model = Unet(features=hyperparameter['hidden_layer'])
        if self.cuda:
            model = model.cuda()
        identifier = hyperparameter['id'] + '_' + hyperparameter['model_index']
        trainer = Trainer(net=model,
                          train_dir=self.train_dir,
                          vali_dir=self.vali_dir,
                          test_dir=self.test_dir,
                          result_dir=self.result_dir,
                          hyperparams=hyperparameter,
                          identifier=identifier,
                          cuda=self.cuda)
        self.trainer = trainer
        print("begin training!")
        trainer.train_model()

if __name__ == "__main__":
    train_dir = r'.\data\training\img'
    vali_dir = r'.\data\validation\img'
    test_dir = r'.\data\testing\img'
    result_dir = r'.\result'
    hyperparameter = dict(
        batch_size=16,
        epochs=200,
        lr=0.001,  ## this param does not work for the cyclic LR
        optimizer='SGD',
        lr_scheduler='CLR',
        CLR_params=[0.01, 0.05, 25],  ## applies only when using cyclic LR
        milestones=[10, 20, 30],  ## applies only when using multistep LR
        gamma=0.75,
        weight_decay=0,
        model='resnet',
        hidden_layer=[32, 64],
        weight=[1, 150, 300],
        model_index='2',
        id='IA_corn_soy',
        crop=['background', 'corn', 'soybean'])
    model = Execution(train_dir, vali_dir, test_dir, result_dir, hyperparameter, cuda=False)
    model.train()