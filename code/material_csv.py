#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:17:29 2020

@author: garethlomax
"""
import pandas as pd
import numpy as np

names = ['iron', 'lead']

density = [7.88e3,11.39e3]

poisson_ratio = [0.3, 0.445]

youngs_modulous = [212e9, 15e9]

dic = {'material' : names, 'density': density, 'poisson_ratio' : poisson_ratio, 'youngs_modulous': youngs_modulous}

df = pd.DataFrame(dic)

