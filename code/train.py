#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:56:47 2020

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

import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import h5py
from .hpc_construct import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda'

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
import pandas as pd

"""training loop for overall training run"""


def wrapper_full(name, optimizer,  structure, loss_func, avg, std, application_boolean, lr = None, epochs = 50, kernel_size = 3, batch_size = 50, dataset_name = 'train_fixed_25.hdf5'):
    """Training wrapper for LSTM encoder decoder models.

    Trains supplied model using train_enc_dec fucntions. Logs model hyperparameters
    and trainging and validation losses in csv training log. Saves the model and
    optimiser state dictionaries after each epoch in order to allow for easy
    checkpointing.

    Parameters
    ----------
    name: str
        filename to save CSV training logs as.
    optimizer: pytorch optimizer
        The desired optimizer needed to train the model
    structure: array of ints
        Structure argument to be passed to lstmencdec. See LSTMencdec for explanation
        of structure format.
    loss_func: pytorch module
        Loss function to be used to calculate training and validation losses.
        The loss should be a CLASS instance of the pytorch loss function, not
        a functinal implementation.
    avg: list of floats
        List of averages for every channel in image sequence loaded. Length should
        equal number of channels in dataset image sequence
    std: list of floats
        List of standard deviations for every channel in image sequence loaded.
        Length should equal number of channels in dataset image sequence.
    application_boolean: list of bools
        List of bools indicating whether standard score normalisation is to be
        applied to each layer. Length should equal number of channels in dataset
        image sequence.
    lr: float
        Learning rate for the optimizer
    epochs: int
        Number of epochs to train the model for
    kernel_size: int
        Size of convolution kernel for the LSTMencoderdecoder
    batch_size: int
        Number of samples in each training minibatch.

    Returns
    -------
    bool:
        indicates if training has been completed.
    """
    f = open(name + ".csv", 'w')
    # open csv file for saving

    # construct model and send to GPU(s)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(LSTMencdec_onestep(structure, 5, kernel_size = kernel_size)).to(device)
    else:
        model = LSTMencdec_onestep(structure, 5, kernel_size = kernel_size).to(device)

    # pass model parameters to optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, amsgrad= True)

    # detail hyperparameters in log file
    f.write("Structure: \n")
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            f.write("{},".format(structure[i,j]))
        f.write("\n") # new line

    f.write("Parameters:\n")
    f.write("optimizer, epochs, learning rate, kernel size \n")

    if lr != None:
        f.write("{},{},{},{},{}\n".format("test", epochs, lr, kernel_size, batch_size))
    else:
        f.write("{},{},{},{},{}\n".format("othertest", epochs, "Default", kernel_size, batch_size))

    f.write("loss_func:\n")
    f.write(loss_func.__repr__() + "\n")

    f.write("optimizer:\n")
    f.write(optimizer.__repr__() + "\n")

    f.write("\n\n\n")
    f.write("TRAINING\n")

    f.close()
    # initialise training and validation datasets.
    train, valid = initialise_dataset_HDF5_full(dataset_name, valid_frac = 0.1, dataset_length = 56413,avg = avg, std = std, application_boolean=application_boolean)

    loss_func = loss_func

    # pass datasets to dataloaders
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True) # implement moving MNIST data input
    valid_loader = DataLoader(valid, batch_size = 2000, shuffle = False) # implement moving MNIST

    for epoch in range(epochs):

        # train the model
        _, loss = train_enc_dec(model, optimizer, train_loader, loss_func = loss_func) # changed

        # save for checkpointing
        torch.save(optimizer.state_dict(), name+str(epoch)+"optimizer.pth")
        torch.save(model.state_dict(), name+str(epoch)+".pth")

        #compute validation at each epoch
        valid_loss = validate(model, valid_loader, loss_func = loss_func) # validation - need to shuffle split.


        f = open(name + ".csv", 'a')
        f.write(str(loss) + "," + str(valid_loss) + "\n")
        f.close()
    return True
