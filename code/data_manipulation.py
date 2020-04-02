#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:21:22 2020

@author: garethlomax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import random
from sklearn.metrics import f1_score, multilabel_confusion_matrix
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda'
import pandas as pd


class TraceDataset(Dataset):
    def __init__(self, trace_path, param_path, transforms = None):
        self.trace_path = trace_path
        self.param_path = param_path
        self.trace = np.load(self.trace_path)
        self.param = pd.read_pickle(self.param_path)
        
    def __getitem__(self, i):
        return self.param.loc[i].to_numpy(), self.trace[i]
    

    
def main():
    trace_path = "../data/Kolsky Bar.npy"
    param_path = "../data/test_grid.pkl"
    
    return TraceDataset(trace_path, param_path)
    
if __name__ == "__main__":
    a = main()
    
        
        