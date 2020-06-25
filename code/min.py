#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:06:20 2020

@author: garethlomax
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
#minimisation polynomial
# import pdb

def rms(array):
   return np.sqrt(np.mean(array ** 2))
    
class poly:
    def __init__(self, x_data, y_data, param_no):
        assert len(x_data) == len(y_data)
        
        self.x = x_data
        self.y = y_data
        self.dim = x_data.shape
        self.prior = None #TODO: evaluate prior here.
        self.param_no = self.dim[1] 
        self.params = np.zeros(1 + self.param_no + self.param_no**2)
        self.error = 1 
        self.tol = 1e-2
        
    def _approx(self,x,param):
        # here x is for single measurement
        # if not isinstance(param, list):
        #     param = [param]
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        obj = param[0]
        obj += np.sum(x * param[1:1+self.param_no])
        
        # print(type(x))
        # print(x)
        for i in range(self.param_no):
            for j in range(self.param_no):
                obj += param[1+self.param_no + i * self.param_no + j] * x[i] * x[j]
               
        return obj       
               
    def approx(self, x, param):
        print(x)
        # print([self._approx(x_, param) for x_ in x][0])
        return np.array([self._approx(x_, param) for x_ in x])
       
    def approx_loss_func(self, param):
        return self.approx(self.x, param) - np.reshape(self.y, (self.x.shape[0],)) #TODO: sort out the reshape
        
    def minimise_loss_func(self, x0):
        print(x0)
        return self._approx(x0, self.params) #- np.reshape(self.y, (50,))
        
        
    def fit(self):
        fit = least_squares(self.approx_loss_func, x0 = self.params)
        self.params = fit.x
        return fit        
        
    def plot(self):
        plt.plot(x, y, 'o')
        plt.plot(x, self.approx(x, self.params), '.')
        # return curve_fit(self.approx, self.x, self.y)
        # optimise least sq here. 
    
    def minimise(self):
        fit = least_squares(self.minimise_loss_func, x0 = self.prior) #TODO: specify where starting location is.
        candidate = fit.x
        return candidate
    
    
    def eval_fem(self, x):
        # Dummy code
        # TODO: 
        y = fem_func(x)
        # y = (x-3)**2 + x + 12
        #TODO: send to fem drivers
        return x, y
    
    
    def fem_update(self):
        n = 0
        error = 1
        prev = 100
        while abs(error) > self.tol and n < 10:
            self.fit()
        
            candidate = self.minimise()
        
            x_new, y_new = self.eval_fem(candidate) #TODO: fix this
            error = abs(y_new - prev)
            prev = y_new
            # we need to stack instead
            x_new.resize((1,2))
            self.x = np.concatenate((self.x, x_new), axis = 0)
            self.y = np.append(self.y, y_new)
        return x_new, y_new, n
        
  
def fem_func(x):
    # return (x[:,0]-3)**2 + x[:,1] + 12
    if len(x.shape) == 1:
        return ((x[0]-3)**2 + x[1] + 12)
    else:
        return (x[:,0]-3)**2 + x[:,1] + 12      

        

        
    
if __name__ == '__main__':
    x = np.random.uniform(high = 3,size = (50,2))
    # x.resize(50,1)
    y = fem_func(x)
    a = poly(x, y, 1)
    a.fit()
    a.plot()
    a.prior = np.zeros(2)
    out = a.minimise()
    a.fem_update()
    
    
    
    