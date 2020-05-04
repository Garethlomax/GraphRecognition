#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:36:55 2020

@author: garethlomax
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class material:
    def __init__(self, name, dist_type, model_type, properties, vals, multiplicative_factors = None):
        self.dist_type = dist_type
        self.properties = properties
        self.vals = vals
        self.model_type = model_type
        if multiplicative_factors is not None:
            self.multiplicative_factors = multiplicative_factors
        else: 
            self.multiplicative_factors = [0.9 for i in properties]
        
        self.mat_prop = dict(zip(properties, vals))
    def gen(self, size = 1):
        
        data = np.random.normal(self.vals, scale = 0.1 * self.vals, size = (size,len(self.vals)))
        return data
    

class material_selection:
    def __init__(self,df):
        self.df = df
        self.mat_list = [material(a.material, 'gaussian', 'elastic', list(a.keys()[1:]), np.array(list(a[1:]))) for index, a in df.iterrows()]
    
    def gen(self, size = 1):
        data_list = []
        
        for i, mat in enumerate(self.mat_list):
            data_list.append(mat.gen(size = size))
        return np.concatenate(data_list)
            

    

        
            
            
        