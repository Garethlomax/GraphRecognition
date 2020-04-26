#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:10:46 2020

@author: garethlomax
"""
import torch
import torch.nn as nn 

device = "cpu"

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # takes entry of dimensons 501 - I think
        input_channels = 1 
        
        self.act = nn.ReLU()
        
        self.l1 = nn.Conv1d(in_channels = input_channels, out_channels = 8,kernel_size = 32, padding = 1, stride = 3)
        self.l2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 16, padding = 1, stride = 4)
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
        
        self.fc1 = nn.Linear(in_features = 512, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 8)
        
        self.fc_logvar = nn.Linear(in_features = 8, out_features = 2)
        self.fc_mu = nn.Linear(in_features = 8, out_features = 2)
        
        self.ifc1 = nn.Linear(in_features = 2, out_features = 8)
        self.ifc2 = nn.Linear(in_features = 8, out_features = 64)
        self.ifc3 = nn.Linear(in_features = 64, out_features = 512)
        
        self.il1 = nn.ConvTranspose1d(in_channels = 128, out_channels = 64, kernel_size = 4, padding = 0 , stride = 2)
        self.il2 = nn.ConvTranspose1d(in_channels = 64, out_channels = 32, kernel_size = 4, padding = 2, stride = 2)
        self.il3 = nn.ConvTranspose1d(in_channels = 32, out_channels = 16, kernel_size = 7, padding = 2, stride = 2)
        self.il4 = nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size = 16, padding = 1, stride = 4)
        self.il5 = nn.ConvTranspose1d(in_channels = 8, out_channels = 1, kernel_size = 32, padding = 1, stride = 3)
        
        self.ibn5 = nn.BatchNorm1d(8)
        self.ibn4 = nn.BatchNorm1d(16)
        self.ibn3 = nn.BatchNorm1d(32)
        self.ibn2 = nn.BatchNorm1d(64)
        self.ibn1 = nn.BatchNorm1d(128)
        
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x):
        h = self.bn1(self.activation(self.l1(x)))
        h = self.bn2(self.activation(self.l2(h)))
        h = self.bn3(self.activation(self.l3(h)))
        h = self.bn4(self.activation(self.l4(h)))
        h = self.bn5(self.activation(self.l5(h)))
        
        #reshape here
        h = h.view(-1, 512)
        
        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        
        logvar = self.fc_logvar(h)
        mu = self.fc_mu(h)
        
        return mu, logvar

    def latent_space(self, mu, log_var):
        if self.training:
          std = torch.exp(0.5*log_var)
          eps = torch.randn_like(std).to(device)
          z = eps.mul(std).add_(mu) 
        else:
          z = mu
        return z
        
    def decoder(self, z):
        
        h = self.activation(self.ifc1(z))
        h = self.activation(self.ifc2(h))
        h = self.activation(self.ifc3(h))
        
        # TODO: check
        h = self.ibn1(h.view(-1,128,4))
        print(h.size())
        
        h = self.ibn2(self.activation(self.il1(h)))
        h = self.ibn3(self.activation(self.il2(h)))
        h = self.ibn4(self.activation(self.il3(h)))
        h = self.ibn5(self.activation(self.il4(h)))
        
        reconstruction = self.sigmoid(self.il5(h))
        
        return reconstruction
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.latent_space(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z, mu, log_var

        
x = torch.randn((10,1,501)).to(device)
model = VAE().to(device)
x_, z, mu, logvar = model(x)
print(x_.size(), z.size(), mu.size(), logvar.size())
        
        