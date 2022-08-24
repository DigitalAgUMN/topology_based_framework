#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chenxi
"""
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from dataset import BuildingDataset
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CyclicLR, MultiStepLR

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def collate_fn(data):
    for i in range(0, len(data)):
        data[i]['label'] = data[i]['label'].sum(axis=0)
    patch = []
    name = []
    image = torch.stack([torch.from_numpy(b['image']) for b in data], 0)
    label = torch.stack([torch.from_numpy(b['label']) for b in data], 0)[:, np.newaxis, :, :]
    patch = patch.append(b['patch'] for b in data)
    name = name.append(b['name'] for b in data)
    return {'image': image, 'patch': patch, 'name':name, 'label':label}

class Trainer(object):
    def __init__(self, net, train_dir, vali_dir, test_dir, result_dir, hyperparams, identifier, cuda=False):
        self.net = net
        self.train_dir = train_dir
        self.vali_dir = vali_dir
        self.test_dir = test_dir
        self.result_dir = result_dir
        self.hyperparams = hyperparams
        self.identifier = identifier
        self.cuda = cuda
        self.opt = hyperparams['optimizer']
        self.learn_rate = hyperparams['lr']
        self.lr_schedule = hyperparams['lr_scheduler']
        self.weight = hyperparams['weight']
        self.wd = hyperparams['weight_decay']
        self.bs = hyperparams['batch_size']
        self.epoch = hyperparams['epochs']
        self.classnames = hyperparams['classname']
        self.train_data = BuildingDataset(dir=self.train_dir, transform=None)
        self.train_loader = utils.data.DataLoader(self.train_data, batch_size=self.bs, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn)
        self.vali_data = BuildingDataset(dir=self.vali_dir, transform=None)
        self.vali_loader = utils.data.DataLoader(self.vali_data, batch_size=self.bs, shuffle=True, num_workers=8)
        self.makefolders()
        self.net.apply(inplace_relu)

    def select_optimizer(self):
        optimizer = None
        if (self.opt == 'Adam'):
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                   lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'RMS'):
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'SGD'):
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                  lr=self.learn_rate, momentum=0.9, weight_decay=self.wd)
        elif (self.opt == 'Adagrad'):
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.learn_rate, weight_decay=self.wd)
        elif (self.opt == 'Adadelta'):
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=self.learn_rate, weight_decay=self.wd)
        return optimizer

    def makefolders(self):
        '''
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        '''
        model_folder = os.path.join(self.result_dir, 'model')
        model_path = os.path.join(model_folder, self.identifier)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_folder = model_folder
        self.model_path = model_path

    def select_scheduler(self, optimizer):
        if self.lr_schedule == 'SLR':
            scheduler = StepLR(optimizer,
                               step_size=4 * len(self.train_loader),
                               gamma=self.hyperparams['gamma'])
        elif self.lr_schedule == 'CLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr=self.hyperparams['CLR_params'][0],
                                 max_lr=self.hyperparams['CLR_params'][1],
                                 step_size_up=self.hyperparams['CLR_params'][2] * len(self.train_loader))
        elif self.lr_schedule == 'MSLR':
            scheduler = MultiStepLR(optimizer,
                                    milestones=self.hyperparams['milestones'],
                                    gamma=self.hyperparams['gamma'])
        return scheduler

    def train_model(self):
        ############ parameters initialization ############
        ###################################################
        torch.backends.cudnn.deterministic = True
        since = time.time()
        optimizer = self.select_optimizer()
        scheduler = self.select_scheduler(optimizer)
        softmax = torch.nn.functional.softmax
        number_of_class = len(self.classnames)
        train_loss = np.zeros([self.epoch])
        vali_loss = np.zeros([self.epoch])
        PA_training, UA_training, PA_validation, UA_validation = {}, {}, {}, {}
        for c in self.classnames:
            PA_training[c] = np.zeros([self.epoch])
            UA_training[c] = np.zeros([self.epoch])
            PA_validation[c] = np.zeros([self.epoch])
            PA_validation[c] = np.zeros([self.epoch])

        ############ network training ############
        for i in range(self.epoch):
            self.net.train()
            accu_loss_training = 0
            for j, sample in enumerate(self.train_loader, 0):
                optimizer.zero_grad()
                image = Variable(sample["image"], requires_grad=False)
                label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor).sum(dim=1)
                weights = torch.FloatTensor(self.hyperparams['weight'])
                if self.cuda:
                    image = image.cuda()
                    label = label.cuda()
                    weights = weights.cuda()
                criterion = nn.CrossEntropyLoss(weight=weights)
                prediction = self.net(image)
                loss = criterion(prediction, label.long())
                accu_loss_training += loss
                filter = torch.ge(torch.max(softmax(prediction, dim=1), dim=1)[0], 0.995)
                if self.cuda:
                    filter = filter.cuda()
                prediction = torch.argmax(softmax(prediction, dim=1), dim=1) * filter
                # cm += confusion_matrix(pred=prediction,
                #                        target=label,
                #                        classes=list(range(number_of_crop+1)))
                cm = confusion_matrix(label.numpy().flatten(), prediction.numpy().flatten())
                loss.backward()
                optimizer.step()
                if self.lr_schedule:
                    scheduler.step()
            for c in range(1, number_of_crop):
                PA_training[i] = cm[c, c] / cm[c, :].sum()
                UA_training[i] = cm[c, c] / cm[:, c].sum()
            accu_loss_training = accu_loss_training.cpu().detach().numpy()
            train_loss[i] = accu_loss_training / len(self.train_loader)

            ############ network validation ############
            ############################################
            self.net.eval()
            accu_loss_vali = 0
            with torch.no_grad():
                for k, sample in enumerate(self.vali_loader, 0):
                    image = Variable(sample["image"], requires_grad=False)
                    label = Variable(sample["label"], requires_grad=False).type(torch.FloatTensor).sum(
                            axis=1)
                    criterion = nn.CrossEntropyLoss()
                    if self.cuda:
                        image = image.cuda()
                        label = label.cuda()
                    prediction = self.net(image)
                    accu_loss_vali += criterion(prediction, label.long())
                    prediction = torch.argmax(softmax(prediction, dim=1), dim=1)
                    cm += confusion_matrix(pred=prediction,
                                           target=label,
                                           classes=list(range(number_of_crop+1)))
                for c in range(1, number_of_crop + 1):
                    PA_vali[i] = cm[c, c] / cm[c, :].sum()
                    UA_vali[i] = cm[c, c] / cm[:, c].sum()
                accu_loss_vali = accu_loss_vali.cpu().detach().data.numpy()
                vali_loss[i] = accu_loss_vali / len(self.vali_loader)
            self.save_model(i)

            elapse = time.time() - since
            print(
                "Epoch:{}/{}\n"
                "training_loss:{}\n"
                "vali_loss:{}\n"
                "Time_elapse:{}\n'".format(
                i + 1, self.epoch,
                round(train_loss[i], 5),
                round(vali_loss[i], 5),
                elapse))

    def save_model(self, epoch):
        torch.save(self.net, os.path.join(self.model_path, self.identifier + '_e_' + str(epoch) + ".pkl"))




