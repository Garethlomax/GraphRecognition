# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:01:53 2020

@author: garethlomax
"""


#quick func to add a timestamp to savefile_name

import datetime

def gen_name(prefix, extension, mode = "datetime"):
    """adds datetime timestamp to filename
    
    if using mode as not None for static time, mode should be a datetime.datetime.now() object"""
    
    if isinstance(mode, datetime.datetime):
    	filename = (prefix + '_' + str(mode.date()) + '_' + str(mode.time()).replace(':', '_').replace('.','_')+ extension)
    else:
    	filename = (prefix + '_' + str(datetime.datetime.now().date()) + "-" + 
        	str(datetime.datetime.now().time()).replace(':', '_').replace('.','_') + extension)
    return filename