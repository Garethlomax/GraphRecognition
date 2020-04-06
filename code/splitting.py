#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:30:03 2020

@author: garethlomax
"""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from gen_filename import *



def split(data, n_splits = 1, fraction = 0.1, random_state = 0):
    """Produces a stratified shuffle split of given dataset

    Wrapper around sklearn functions to produce stratified shuffle split of given
    dataset

    """
    dummy_array = np.zeros(len(data))
    split = StratifiedShuffleSplit(n_splits, test_size = fraction, random_state = 0)
    generator = split.split(torch.tensor(dummy_array), torch.tensor(dummy_array))
    return [(a,b) for a, b in generator][0]


def ttv_split(data, train, valid, test, random_state = 0, save_name = False):
    """produces stratified shuffle split of data set
    
    Wrapper around split and sklearn functions to produce split dataset
    
    Parameters
    ----------
    
    data: array
        Array of data to be split
        
    train: float
        float of decimal 0 - 1 of the total data set to be used for a training 
        set, before a validtion set is extracted
        
    valid: float
        float of decimal 0 - 1 of the split training data set to be used for a
        validation set
        
    test: float
        float of decimal 0 - 1 of the total data set to be used for a test 
        set
        
    Returns
    -------
    
    Array:
        array of indices to be used for splitting dataset
    """
    assert (train + test == 1), "train and test fractions must account for all of the dataset"
    
    #splitting inital dataset
    train_index, test_index = split(data, fraction = test, random_state = random_state)
    # print(len(train_index))
    # print(len(test_index))
    
    train_index_split, valid_index = split(train_index, fraction = valid, random_state= random_state)
    # print(len(train_index_split))
    # print(len(valid_index))
    
    valid_index = train_index[valid_index]

    train_index = train_index[train_index_split] 
    
    return train_index, valid_index, test_index
    
    
    
    
    
    
    