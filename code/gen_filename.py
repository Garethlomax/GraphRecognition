
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:01:53 2020

@author: garethlomax
"""


#quick func to add a timestamp to savefile_name

import datetime

def gen_name(prefix, extension, mode = "datetime"):
    """adds datetime timestamp to filename"""
    
    filename = (prefix + '_' + str(datetime.datetime.now().date()) + "-" + 
        str(datetime.datetime.now().time()).replace(':', '_').replace('.','_') + extension)
    return filename