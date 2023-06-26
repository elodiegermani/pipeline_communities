#!/usr/bin/env python

import torch.nn as nn
import numpy as np


class Classifier3D(nn.Module):

    def __init__(self):
        super(Classifier3D, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1, stride=2)
        self.norm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2)
        self.norm2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=2)
        self.norm3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1, stride=2)
        self.norm4 = nn.BatchNorm3d(512)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 =  nn.Linear(512*3*4*3, 24)

    ########
    #
    #  Constructed Classifier
    #
    ########
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)

        return x