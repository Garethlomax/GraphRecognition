#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:10:46 2020

@author: garethlomax
"""

import torch.nn as nn 

class VAE(nn.module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # takes entry of dimensons 501 - I think
        input_channels = 1 
        
        self.act = nn.ReLU()
        
        self.l1 = nn.Conv1d(in_channels = input_channels, out_channels = 8,kernel_size = 32, padding = 1, stride = 3)
        self.l2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 16, paddng = 1, stride = 4)
        self.l3 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 7, padding = 2, stride = 2)
        self.l4 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 4, padding = 2, stride = 2)
        self.l5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 4, padding = 0, stride = 2)
        
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        
        # now flatten the tensor
        # can use max unpool1d or 2d.
        
        self.fc1 = nn.Linear(in_features = 512, output_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 8)
        
        self.fc_logvar = nn.Linear(in_features = 8, out_features = 2)
        self.fc_mu = nn.Linear(in_features = 8, out_features = 2)
        
        self.ifc1 = nn.Linear(in_features = 2, out_features = 8)
        self.ifc2 = nn.Linear(in_features = 8, out_features = 64)
        self.ifc3 = nn.Linear(in_features = 64, outfeatures = 512)
        
        self.il1 = nn.ConvTranspose1d(in_channels = 128, out_channels = 64, kernel_size = 4, padding = 0 , stride = 2)
        self.il2 = nn.ConvTranspose1d(in_channels = 64, out_channels = 32, kernel_size = 4, padding = 2, stride = 2)
        self.il3 = nn.ConvTranspose1d(in_channels = 32, out_channels = 16, kernel_size = 7, padding = 2, stride = 2)
        self.il4 = nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size = 16, paddin = 1, stride = 4)
        self.il5 = nn.ConvTranspose1d(in_channels = 8, out_channels = 1, kernel_size = 32, padding = 1, stride = 3)
        
        self.bn5 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        
        
        