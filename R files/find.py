#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:08:46 2022

@author: Rohith
"""


import numpy as np 
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAnnulus


#%%
class image:
    
    ''' This class will find the brightest source in the image'''
    
    def __init__(self,array):
        self.array = array 
        self.base_arr = np.ones(self.array.shape, dtype = int)

    
        
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
    
    
    def aperture(self):
        x = self.find()[1][0]
        y = self.find()[1][1]
    
        loc_tup = []
        for i in range(len(x)):
           l =(x[i],y[i]) 
           loc_tup.append(l)
        
        aperture = CircularAperture(loc_tup, r=3)
        annulus_aperture = CircularAnnulus(loc_tup, r_in=4, r_out=5)
        apers = [aperture, annulus_aperture]


        phot_table = aperture_photometry(self.array, apers)

        for col in phot_table.colnames:
            phot_table[col].info.format = '%.8g'  # for consistent table output

        return phot_table


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
new_array = np.random.randint(101, size=(10,10))

# print(new_array)
# Max = np.amax(new_array)
# print(Max)
# np.where(new_array == Max)
print(new_array)
A = image(new_array)
info = catalogue(new_array)
print(A.aperture())
# info.Next()


#%%
from matplotlib import pyplot

x0 = 4; a = 5  # x center, half width                                       
y0 = 2; b = 3  # y center, half height                                      
x = np.linspace(-10, 10, 100)  # x values of interest
y = np.linspace(-5, 5, 100)[:,None]  # y values of interest, as a "column" array
ellipse = ((x-x0)/a)**2 + ((y-y0)/b)**2   # True for points inside the ellipse

pyplot.imshow(ellipse, extent=(-10, 10, -10, 10), origin="lower", cmap='RdGy')  # Plot

# %%



positions = [(5,5),(6,6)]

aperture = CircularAperture(positions, r=3)
annulus_aperture = CircularAnnulus(positions, r_in=3, r_out=4)
apers = [aperture, annulus_aperture]

data = np.random.randint(10, size=(10,10))
phot_table = aperture_photometry(data, apers)

for col in phot_table.colnames:
    phot_table[col].info.format = '%.8g'  # for consistent table output
print(data)
print(phot_table)














