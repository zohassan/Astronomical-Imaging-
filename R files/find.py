#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:08:46 2022

@author: Rohith
"""


import numpy as np 

#%%
class image:
    
    ''' This class will find the brightest source in the image'''
    
    def __init__(self,array):
        self.array = array 
        self.base_arr = np.full(self.array.shape, 1, dtype=int)

    
        
    def find(self):
        Max = np.amax(self.array)
        location = np.where(self.array == Max)     
        return Max, location 

    
    def mask(self):
        x = self.find()[1][0]
        y = self.find()[1][1]
        
        for i in range(len(x)):
            self.array[x[i]][y[i]] = 0
        return 


#%%


class catalogue:
    
    def __init__(self,array):
        self.image = image(array)
        self.info = []
        
        
    def Next(self):
        x = self.image.find()
        self.info.append(x)
        self.image.mask()
        


# %%
new_array = np.random.randint(5, size=(5,3))

# print(new_array)
# Max = np.amax(new_array)
# print(Max)
# np.where(new_array == Max)
print(new_array)
A = image(new_array)
info = catalogue(new_array)
info.Next()
print(A.array)
info.Next()
print(A.array)
info.Next()
print(A.array)

#%%
from matplotlib import pyplot

x0 = 4; a = 5  # x center, half width                                       
y0 = 2; b = 3  # y center, half height                                      
x = np.linspace(-10, 10, 100)  # x values of interest
y = np.linspace(-5, 5, 100)[:,None]  # y values of interest, as a "column" array
ellipse = ((x-x0)/a)**2 + ((y-y0)/b)**2   # True for points inside the ellipse

pyplot.imshow(ellipse, extent=(-10, 10, -10, 10), origin="lower", cmap='RdGy')  # Plot