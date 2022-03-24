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

if __name__ == "__main__":
    # set the test input
    unet = Unet()
    images = Variable(torch.rand(4, 3, 256, 256))
    prediction = unet(images)
