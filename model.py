#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:00:06 2018

@author: huijian
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


def bn_conv_relu(in_channels, out_channels):
    """
    usually features = [64,96]
    """
    conv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

    return conv


def std_upconv(features, top=False):
    """
    usually features = [64,96]
    """
    if top:
        upconv = nn.Sequential(
            nn.BatchNorm2d(features[0] * 2),
            nn.Conv2d(features[0] * 2, features[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(features[1]),
            nn.Conv2d(features[1], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(features[0], 4, kernel_size=1, stride=1, padding=0))

        return upconv

    upconv = nn.Sequential(
        nn.BatchNorm2d(features[0] * 2),
        nn.Conv2d(features[0] * 2, features[1], kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(features[1]),
        nn.Conv2d(features[1], features[0], kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(features[0]),
        nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2, padding=0),
        nn.ReLU()
    )
    return upconv


class Unet(nn.Module):
    def __init__(self, in_channels=4, features=[64, 96]):
        super(Unet, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv12 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv22 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv23 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv32 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv33 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv42 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv43 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv51 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv52 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv53 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv61 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv62 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv63 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(features[0]),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(features[0]),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.upconv1 = std_upconv(features)
        self.upconv2 = std_upconv(features)
        self.upconv3 = std_upconv(features)
        self.upconv4 = std_upconv(features)
        self.upconv5 = std_upconv(features)
        self.upconv6 = std_upconv(features, top=True)

    def forward(self, x):
        x = self.preconv(x)  # (b,3,h,w) -> (b,64,h,w) [b, 64, 192, 192]
        saved_part = []

        x = self.conv11(x) # [b, 64, 192, 192]
        x = self.conv12(x) # [b, 64, 192, 192]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 96, 96]
        # print(1)
        x = self.conv21(x) # [b, 64, 96, 96]
        x = self.conv22(x) # [8, 64, 96, 96]
        saved_part.append(x)
        # x = self.conv23(x) # [8, 16, 64, 64]
        x = self.maxpool1(x) # [8, 16, 32, 32]
        # print(2)
        x = self.conv31(x) # [8, 16, 32, 32]
        x = self.conv32(x) # [8, 16, 32, 32]
        saved_part.append(x)
        x = self.conv33(x) # [8, 16, 32, 32]
        x = self.maxpool1(x) # [8, 16, 16, 16]
        # print(3)
        x = self.conv41(x) # [8, 16, 16, 16]
        x = self.conv42(x) # [8, 16, 16, 16]
        saved_part.append(x)
        x = self.conv43(x) # [8, 16, 16, 16]
        x = self.maxpool1(x) # [8, 16, 8, 8]
        # print(4)
        x = self.conv51(x) # [8, 16, 8, 8]
        x = self.conv52(x) # [8, 16, 8, 8]
        saved_part.append(x)
        x = self.conv53(x) # [8, 16, 8, 8]
        x = self.maxpool1(x) # [8, 16, 4, 4]
        # print(5)
        x = self.conv61(x) # [8, 16, 4, 4]
        x = self.conv62(x) # [8, 16, 4, 4]
        saved_part.append(x)
        x = self.conv63(x) # [8, 16, 4, 4]
        x = self.maxpool1(x) # [8, 16, 2, 2]
        # print(5)
        x = self.bottom(x) # [8, 16, 4, 4]

        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [8, 32, 4, 4]
        # print(0)
        x = self.upconv1(x)  # [8, 16, 8, 8]
        # print(1)
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [8, 32, 8, 8]
        x = self.upconv2(x) # [8, 16, 16, 16]
        # print(2)
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [8, 32, 16, 16]
        x = self.upconv3(x) # [8, 16, 32, 32]
        # print(3)
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [8, 32, 32, 32]
        x = self.upconv4(x) # [8, 16, 64, 64]
        # print(4)
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [8, 32, 64, 64]
        x = self.upconv5(x) # [8, 16, 128, 128]

        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [8, 32, 128, 128]
        x = self.upconv6(x) # [8, 1, 128, 128]
        return x

class Unet_3(nn.Module):
    def __init__(self, in_channels=4, features=[64, 96]):
        super(Unet_3, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv12 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv22 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv32 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(features[0]),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(features[0]),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.upconv1 = std_upconv(features)
        self.upconv2 = std_upconv(features)
        self.upconv3 = std_upconv(features, top=True)

    def forward(self, x):
        x = self.preconv(x)  # (b,3,h,w) -> (b,64,h,w) [b, 64, 192, 192]
        saved_part = []

        x = self.conv11(x) # [b, 64, 192, 192]
        x = self.conv12(x) # [b, 64, 192, 192]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 96, 96]
        x = self.conv21(x) # [b, 64, 96, 96]
        x = self.conv22(x) # [b, 64, 96, 96]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 48, 48]
        x = self.conv31(x) # [b, 64, 48, 48]
        x = self.conv32(x) # [b, 64, 48, 48]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 24, 24]
        x = self.bottom(x) # [b, 64, 48, 48]

        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 48, 48]
        x = self.upconv1(x)  # [b, 64, 96, 96]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 96, 96]
        x = self.upconv2(x) # [b, 64, 192, 192]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 192, 192]
        x = self.upconv3(x) # [b, 1, 192, 192]

        return x

class Unet_4(nn.Module):
    def __init__(self, in_channels=1, features=[64, 96]):
        super(Unet_4, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv11 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv12 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv13 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv22 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv23 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv32 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv33 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv42 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.conv43 = bn_conv_relu(in_channels=features[0], out_channels=features[0])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottom = nn.Sequential(
            nn.BatchNorm2d(features[0]),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(features[0]),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.upconv1 = std_upconv(features)
        self.upconv2 = std_upconv(features)
        self.upconv3 = std_upconv(features)
        self.upconv4 = std_upconv(features, top=True)

    def forward(self, x):
        x = self.preconv(x)  # (b,3,h,w) -> (b,64,h,w) [b, 64, 192, 192]
        saved_part = []

        x = self.conv11(x) # [b, 64, 192, 192]
        x = self.conv12(x) # [b, 64, 192, 192]
        x = self.conv13(x)  # [b, 64, 192, 192]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 96, 96]
        x = self.conv21(x) # [b, 64, 96, 96]
        x = self.conv22(x) # [b, 64, 96, 96]
        x = self.conv23(x)  # [b, 64, 96, 96]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 48, 48]
        x = self.conv31(x) # [b, 64, 48, 48]
        x = self.conv32(x) # [b, 64, 48, 48]
        x = self.conv33(x)  # [b, 64, 48, 48]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 24 , 24]
        x = self.conv41(x) # [b, 64, 24, 24]
        x = self.conv42(x) # [b, 64, 24, 24]
        x = self.conv43(x)  # [b, 64, 24, 24]
        saved_part.append(x)
        x = self.maxpool1(x) # [b, 64, 12, 12]
        x = self.bottom(x) # [b, 64, 12, 12]

        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 24, 24]
        x = self.upconv1(x)  # [b, 64, 48, 48]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 48, 48]
        x = self.upconv2(x) # [b, 64, 96, 96]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 96, 96]
        x = self.upconv3(x) # [b, 64, 192, 192]
        tmp = saved_part.pop()
        x = torch.cat((x, tmp), 1) # [b, 128, 192, 192]
        x = self.upconv4(x) # [b, 1, 192, 192]
        return x

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = nn.ReLU(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.ReLu(Y)





if __name__ == "__main__":
    # set the test input
    unet = Unet()
    images = Variable(torch.rand(4, 3, 256, 256))
    prediction = unet(images)
