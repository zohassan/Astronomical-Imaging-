#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:00:38 2022

@author: Rohith
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

# %%
hdulist = fits.open('A1_mosaic.fits')
counts = hdulist[0].data
counts = counts.flatten()

cnts = []

for i in range(len(counts)):
    if counts[i] <3500 and counts[i] > 3350:
        cnts.append(counts[i])
#%%

bins = len(cnts)
n, bins, t = plt.hist(cnts,247,density= True)
plt.xlim(3300,3550)

mu, sigma = stats.norm.fit(cnts)
best_fit_line = stats.norm.pdf(bins, mu, sigma)
plt.plot(bins,best_fit_line)


#%%

n, bins, t = plt.hist(cnts,len(cnts),density= True)
plt.xlim(3300,3550)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

x = bins[1:]
popt,pcov = curve_fit(gaus,x,n, p0 =[340000,3420,17.55])

plt.plot(x,gaus(x,popt[0],popt[1],popt[2]),label='fit')

# %%

