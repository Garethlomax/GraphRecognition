#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:04:11 2020

@author: garethlomax
"""
import numpy as np
from splitting import ttv_split
from data_manipulation import TraceDataset

def make_datasets(pickle_path, numpy_path, train, valid, test, dataset = TraceDataset):
    """Produces dataset from train, valid and test string locations from numpy 
    strings"""
    
    #load 
    train = np.load(train)
    valid = np.load(valid)
    test  = np.load(test)
    
    
    train_dset = dataset(pickle_path, numpy_path,index_mapping = train)
    valid_dset = dataset(pickle_path, numpy_path,index_mapping = valid)
    test_dset = dataset(pickle_path, numpy_path,index_mapping = test)
    
    return train_dset, valid_dset, test_dset

    
    
