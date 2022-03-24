#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:16:55 2022

@author: Rohith
"""

# %% Documentation 

'''
paste: 
    from testdata import data, bkgdata 
    
to import the test data files 
'''

# %% Modules 
from photutils.datasets import make_100gaussians_image
import numpy as np 

# %% Background Only 

bkgdata = np.random.normal(3418,12,(4611,2570))

# %% 100 Gaussians

data = make_100gaussians_image()

