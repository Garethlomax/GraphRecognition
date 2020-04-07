#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:00:02 2020

@author: garethlomax
"""


from prod_dataset import make_datasets
from data_manipulation import TraceDataset
from torch.utils.data import TensorDataset, DataLoader, Dataset
from train import wrapper_vae


#TODO: write paths to dataset 

pickle_path= None
numpy_path = None
train = None
valid = None 
test = None 
dataset = TraceDataset
batch_size = None

if __name__ == "__main__":
    
    train, valid, test = make_datasets(pickle_path, numpy_path, train, valid, test, dataset)
    
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid, batch_size = batch_size, shuffle = False)
    
    
    
    