#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:15:40 2020

@author: garethlomax
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as ss


data = np.load("../data/Model-1_2020-04-25-22_36_54_211954.npy")

df = pd.read_pickle("../data/test_grid_2020-04-26-09_34_50_413400.pkl")



# down = ss.decimate(data[0,:,1], 20)

def reduce_data(data, downsampling_ratio, cutoff, save = False, filename = "dset.npy"):
    """downsample traces in dataset to produce into another"""
    data = ss.decimate(data[:,:,1], downsampling_ratio)
    data = data[:,:cutoff]
    if save:
        np.save(filename, data)
    return data
    

def add_fail(data, df):
    """adds boolean checking if simulation failed"""
    fail_boolean = [np.array_equal(data[i], np.zeros((10001,2))) for i in range(len(data))]
    
    df["completion_failiure"] = fail_boolean
    

def param_plots(data, df):  
    add_fail(data, df)
    
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    xs = df['density']
    ys = df['youngs_modulous']
    zs = df['poisson_ratio']
    c = ['red' if i else 'green' for i in df['completion_failiure']]
    
    ax.scatter(xs, ys, zs, s=50, alpha=0.6,edgecolors='w', color = c)
    
    ax.set_xlabel('density')
    ax.set_ylabel('youngs_modulous')
    ax.set_zlabel('poisson_ratio')
    
    plt.show()
    
    
    cols = ['density', 'poisson_ratio', 'youngs_modulous', 'vals','completion_failiure']
            
    pp = sns.pairplot(data=df[cols], 
                      hue='completion_failiure', # Look here!
                      size= 1.8, aspect=1.8, 
                      plot_kws=dict(edgecolor="black", linewidth=0.5))
    fig = pp.fig 
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
    
    
    