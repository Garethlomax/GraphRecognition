#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:17:55 2020

@author: garethlomax
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as ss

data = np.load("../data/iron.npy")

for i in range(len(data)):
    plt.plot(data[i,:,1])
    