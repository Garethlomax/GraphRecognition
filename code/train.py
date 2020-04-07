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

from gen_filename import gen_name

import datetime

"""training loop for overall training run"""

# put printing into seperate function 

def validate(model, dataloader, loss_func, verbose = False):
    """validation routine for validation of VAE models.
    """
    i = 0
    total_loss = 0 
    model.eval()
    
    for x in dataloader:
        with torch.no_grad():
            x = x.to(device)
            prediction = model(x)
            loss = loss_func(prediction, x)
            total_loss += loss
            i += 1
    return total_loss / i 
    


def train_enc_dec(model, optimizer, dataloader, loss_func = nn.MSELoss(), verbose = False):
    """Training function for encoder decoder models.

    Parameters
    ----------
    model: pytorch module
        Input model to be trained. Model should be end to end differentiable,
        and be inherited from nn.Module. model should be sent to the GPU prior
        to training, using model.cuda() or model.to(device)
    optimizer: pytorch optimizer.
        Pytorch optimizer to step model function. Adam / AMSGrad is recommended
    dataloader: pytorch dataloader
        Pytorch dataloader initialised with hdf5 averaged datasets
    loss_func: pytorch loss function
        Pytorch loss function
    verbose: bool
        Controls progress printing during training.

    Returns
    -------
    model: pytorch module
        returns the trained model after one epoch, i.e exposure to every piece
        of data in the dataset.
    tot_loss: float
        Average loss per sample for the training epoch
    """
    i = 0
    model.train()
    # model now tracks gradients
    tot_loss = 0
    for x, y in dataloader:
        x = x.to(device) # Copy image tensors onto GPU(s)
        y = y.to(device)
        optimizer.zero_grad()
        # zeros saved gradients in the optimizer.
        # prevents multiple stacking of gradients

        prediction = model(x)

        if verbose:
            print(prediction.shape)
            print(y.shape)
            
        loss = loss_func(prediction[:,0,0], y)

        # differentiates model parameters wrt loss
        loss.backward()

        optimizer.step()
        # steps forward model parameters

        tot_loss += loss.item()

        if verbose:
            print("BATCH:")
            print(i)
        i += 1

        if verbose:
            print("MSE_LOSS:", tot_loss / i)
        tot_loss /= i
    return model, tot_loss # trainloss, trainaccuracy

def train_vae(model, optimizer, dataloader, loss_func = nn.MSELoss(), verbose = False):
    """Training function for training VAEs
    
    trains through one full epoch comparing reconstruction loss"""
    
    model.train()
    i = 0 
    total_loss = 0
    for x in dataloader:
        x = x.to(device)
        optimzier.zero_grad()
        prediction = model(x)
        if verbose:
            print(x.shape)
            print(prediction.shape)
        # reconstruction loss
        loss = loss_func(prediction, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1
    
    if verbose:
        print("Epoch Loss:")
        print(total_loss)
    return model, total_loss

def wrapper_vae(model_name, optimizer, loss_func, lr = None, epochs = 50, **kwargs):
    """Wrapper for VAE training

    Parameters
    ----------
    model_name : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    loss_func : TYPE
        DESCRIPTION.
    lr : TYPE, optional
        DESCRIPTION. The default is None.
    epochs : TYPE, optional
        DESCRIPTION. The default is 50.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # 
    time = datetime.datetime.now()
    
    for epoch in range(epochs):
        
        # trainig
        # validation 

        # loss 
        # loss logging 
        # saving 
        
        model, loss = train_vae(model, optimizer, train_loader, loss_func) # do we need to reassign model at this step?? 
        
        validation_loss = validate(model, validation_loader, loss_func)

        model_filename = gen_name(model_name + "_epoch_" + str(epoch) + "_time_", ".pth", mode = time)
        optimizer_filename = gen_name(model_name + "_epoch_" + str(epoch) + "_time_", ".pth", mode = time)
        
        torch.save(optimizer.state_dict(), optimizer_filename)
        torch.save(model.state_dict(), model_filename)
        
    return True
        
    
    
    
    
    

def wrapper_full(name, model, model_spec, optimizer,  structure, loss_func, avg, std, application_boolean, lr = None, epochs = 50, kernel_size = 3, batch_size = 50, dataset_name = 'train_fixed_25.hdf5'):
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
        model = nn.DataParallel(model(*model_spec)).to(device)
    else:
        model = model(*model_spec).to(device) # using * to upack list of model specifications 

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
    
    
    
    #splitting apart - change this workijng method. 
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
