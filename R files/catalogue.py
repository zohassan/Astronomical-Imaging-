#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:46:14 2022

@author: Rohith
"""

#%% modules 
'''These are the modules required for this code to run. A lot of these are 
imported modules from astropy and photutils to detect sources and backgrounds 
and catalgoue their fluxes 
'''

from astropy.io import fits
import numpy as np
from astropy.convolution import Gaussian2DKernel
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.segmentation import deblend_sources
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import SourceCatalog
from photutils.utils import calc_total_error
import math
import matplotlib.pyplot as plt
#%% Data 
'''
This cell opens the fits file, extracts the data and coverts the data in the
2D array from integers to floats
'''

hdulist = fits.open('A1_mosaic.fits')
counts = hdulist[0].data
counts = counts + 0.0


# %% Fake Data
'''
Fake background data to test the code and observe if any sources are detected.
'''
x = np.random.normal(3418,12,(4611,2570))
counts = x 

# %% Mask 
'''
This cell opens as 2D array with the same shape as the data to mask bright so-
urces within the data
'''
mask = np.genfromtxt('base_masked_bleeding.csv',delimiter = ',')
mask = mask <1 # This is to change the mask values from false to true 
# %% Mask pt2

t = np.where(counts > 50000)
mask[t] = True


# %% Background Removal 
'''
This cell estimates the median background across the array and then uses this 
estimate to calculate background levels between different sections of the arr-
ay. A block size 53x257 was chosen to calculate the background for an integer 
number of blocks.

The background must be removed from the array for the next step of the process
'''
bkg_estimator = MedianBackground()
bkg = Background2D(counts, (53, 257), filter_size=(3, 3),bkg_estimator=bkg_estimator,mask = mask)
counts -= bkg.background  # subtract the background
threshold = 3. * bkg.background_rms  # above the background

# %% Detect sources 
'''
This cell detects all the sources in the 2D array once the background is 
removed. The kernel smooths the image and removes the noise before thresholdi-
ng. This maximises the detectability of sources similar to the kernel.

detect_sources then finds pixels which have 'npixels' number of connected pix-
els which are above the threshold. In this case this is 5. detect_sources then 
creates a SegmentationImage which is 2D array of 0s for background and integer
values for pixels which contain a source. 

debelend_sources deals with sources which are very close together. It uses key
words such as 'nlevels' and 'contrast' to distinguish sources. nlevels is the 
number of multi-thresholding levels to use. contrast is the fraction of the 
total source flux that a local peak must have to be considered as a separate 
object.
'''

sigma = 4.0 * gaussian_fwhm_to_sigma  # FWHM = 4.
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize()
npixels = 5
segm = detect_sources(counts, threshold, npixels=npixels, kernel=kernel,mask = mask)
# segm_deblend = deblend_sources(counts, segm, npixels=npixels,kernel=kernel, nlevels=32,contrast=0.001)

# %% Source Catalogue 
'''
The first two lines calculate the possible error in the flux of the sources.
It calculates the background only error plus Poissoin noise due to individual 
sources. Effective gain is the exposure time of this image. 

cat then produces a SourceCatalog object which which can then produce a table 
of the information about the sources. Such as the positions but also the kron
flux and the segment flux of the sources and their respective errors. The kron
flux is the flux from elliptical apertures. The segment flux is the flux from
pixels segmented as the sources. 
'''

effective_gain = 54.1
error = calc_total_error(counts, bkg.background_rms, effective_gain)


# cat = SourceCatalog(counts, segm_deblend, error = error)

cat = SourceCatalog(counts, segm, error = error)
tbl = cat.to_table()
# tbl['xcentroid'].info.format = '.2f'  # optional format
# tbl['ycentroid'].info.format = '.2f'
# tbl['kron_flux'].info.format = '.2f'
print(tbl)

# %% Visualising the fit 
'''
This produces a visualisation of the sources with the first plot being the ac-
tual sources themselves and the second plot being the segmented sources with 
kron apertures 
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
norm = simple_norm(counts, 'sqrt')
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(20, 25))
ax1.imshow(counts, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Data')
cmap = segm_deblend.make_cmap(seed=123)
ax2.imshow(segm_deblend, origin='lower', cmap=cmap,interpolation='nearest')
ax2.set_title('Segmentation Image')
cat.plot_kron_apertures((2.5, 1.0), axes=ax1, color='white', lw=1)
cat.plot_kron_apertures((2.5, 1.0), axes=ax2, color='white', lw=1)

# %% Counts vs Magnitude 

'''
This cell produces all the magnitudes of the sources from their fluxes, the 
number counts from those magnitudes and the errors in the number counts. The 
cell does this for both the kron flux and the segmented flux. The magnitude 
from kron flux is simply M whereas from segmented flux its segM. Similar
notation is used throughout.

Number counts was found by finding the number of sources which had a magnitude
less then a given magnitude for all the found magnitudes. Initially an error
was calculated using the error in magnitudes but a later a Poissoin error cal-
culation approach was used instead. 
'''

M = [] # magnitude list
Merr = [] # error in mag
Merrf = [] #  a fixed list of errors 
for i in range(len(tbl)):
    if tbl[i]['kron_flux'] == float('NaN'):
        continue

    k = tbl[i]['kron_flux']
    kerr = tbl[i]['kron_fluxerr']
    m = 25.3 - (2.5*np.log10(k))
    merr = 0.5*(np.log10(k+kerr) - np.log10(k-kerr))
    merrf = 0.06
    M.append(m)
    Merr.append(merr)
    Merrf.append(merrf)
    
    
M = [x for x in M if math.isnan(x) == False] # Removes 'nan' values
Merr = [x for x in Merr if math.isnan(x) == False]
M = np.array(M)
Merr = np.array(Merr)
Merrf = np.array(Merrf)

# M = M[M > 19.5]
# M = M[M < 22]

segM = [] #a lis of magnitudes for segment flux 
segMerr = []

for i in range(len(tbl)):
    k = tbl[i]['segment_flux']
    kerr = tbl[i]['segment_fluxerr']
    m = 25.3 - (2.5*np.log10(k))
    merr = np.log10(k+kerr) - np.log10(k-kerr)
    segM.append(m)
    segMerr.append(merr)

segM = [x for x in segM if math.isnan(x) == False]
segMerr = [x for x in segMerr if math.isnan(x) == False]
segM = np.array(segM)
segMerr = np.array(segMerr)

# segM = segM[segM > 19.5]
# segM = segM[segM < 17]

N = [] #  a number count for kron mag
Nerr = []
Nerrf = []
for i in range(len(M)):
    k = M[M<=M[i]]
    k_p = M[M < (M[i]+Merr[i])]
    k_m = M[M < (M[i]-Merr[i])]
    k_pf = M[M < (M[i]+Merrf[i])]
    k_mf = M[M < (M[i]-Merrf[i])]
    n = len(k)
    n_err = len(k_p)-len(k_m)
    n_errf = len(k_pf)-len(k_mf)
    N.append(n)
    Nerr.append(n_err)
    Nerrf.append(n_errf)
    
N = np.array(N)
Nerr = np.array(Nerr)
Nerrf = np.array(Nerrf)
# N = N[N > 0]
    

segN = [] # a number count for segment mag 
segNerr = []
for i in range(len(segM)):
    k = segM[segM<=segM[i]]
    # k_p = segM[segM < (segM[i]+segMerr[i])]
    # k_m = segM[segM < (segM[i]-segMerr[i])]
    n = len(k)
    # n_err = len(k_p)-len(k_m)
    segN.append(n)
    # segNerr.append(n_err)
    
segN = np.array(segN)
segNerr = np.array(segNerr)
# N = N[N > 0]


# these are the error in the logs of N using the given magnitudes 
lnN = np.log10(N)
lnNerr = np.log10(Nerr)
lnNerrf = np.log10(Nerrf)
lnsegN = np.log10(segN)
lnsegNerr = np.log10(segNerr)

# These are the errors due to Poissoin counting
yerr = 1/((np.sqrt(N))*np.log(10)) 
segyerr = 1/((np.sqrt(segN))*np.log(10))




# %% Plots 

'''
This cell fits a stright line to log10(N) vs M and then plots this straight 
line with the points found. Plots for both kron mag and segment mag are given.
Poissoin errors are used. 
'''



def limarray(array1,array2,limvalue): 
    '''
    This function limits the values of magnitude when fittting to remove the
    effects of completeness on the fits 
    
    Parameters
    ----------
    array1 : The magnitude array you want to limit
    array2 : The log of number count array 
    limvalue : the value you want to limit to 

    Returns
    -------
    array1 : the reduced magnitude array
    array2 : the reduced log number count array

    '''
    X = []
    for i in range(len(array1)): 
        if array1[i] > limvalue:
            x = np.where(array1 == array1[i])
            X.append(x)
        
    array1 = np.delete(array1,X)
    array2 = np.delete(array2,X)
    return array1, array2

         

# plt.plot(M,lnN,'x')
plt.errorbar(M,lnN,yerr=yerr, fmt ='x')
plt.errorbar(segM,lnsegN,yerr=segyerr,fmt = 'o')

fittingM , fittingN = limarray(M,lnN,18.6)
fittingsegM, fittingsegN = limarray(segM,lnsegN,17)


fit , cov = np.polyfit(fittingM,fittingN,1, cov = True)

segfit,segcov = np.polyfit(fittingsegM,fittingsegN,1,cov=True)

plt.plot(M, M*fit[0] + fit[1])
plt.plot(segM, (segM*segfit[0])+segfit[1])

# %% cumulative bins

Msc = np.linspace(10,21,100)
Mc = np.linspace(10,21,100)
Nc = []
segNc = []

for i in range(len(Mc)):
    k = M[M < Mc[i]]
    sk = segM[segM < Msc[i]]
    sn = len(sk)
    n = len(k)
    Nc.append(n)
    segNc.append(sn)
    
lnNc = np.log10(Nc)
lnsegNc = np.log10(segNc)

yerrc = 1/((np.sqrt(Nc))*np.log(10))
yerrsc = 1/((np.sqrt(segNc))*np.log(10))


plt.errorbar(Mc,lnNc,yerr=yerrc,fmt='x',color='blue',mew = 1,ms=6,label ='lnN')
# plt.errorbar(Msc,lnsegNc,yerr=yerrc,fmt='x')

fitMc,fitlnNc = limarray(Mc,lnNc,Mc[70])
fitMsc, fitsegNc = limarray(Mc,lnsegNc,17)
fitc,covc = np.polyfit(fitMc,fitlnNc,1,cov=True)
fitsc,covsc =np.polyfit(fitMsc,fitsegNc,1,cov=True)
plt.plot(Mc, (Mc*fitc[0])+fitc[1],color='red',lw=1.5,label = 'Natural log fit')
# plt.plot(Msc, (Msc*fitsc[0])+fitsc[1])

binchange=[]
for i in range(len(lnNc)):
    x = Nc[i]-Nc[i-1]
    binchange.append(x)

# %%


plt.errorbar(Mc,lnNc,yerr=yerrc,fmt='x',color='blue',mew = 1.3,ms=10,label ='lnN')
plt.plot(Mc[0:85], (Mc[0:85]*fitc[0])+fitc[1],color='red',lw=3,label = 'Natural log fit')

plt.xlabel('Magnitude',fontsize = 17.5)
plt.ylabel('Natural Log of Binned Number Counts',fontsize = 15)
plt.legend(fontsize = 17.5)
plt.title('A1_Mosaic',fontsize = 20)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10,5)
plt.grid(True)
plt.tick_params(axis = 'both',labelsize =13)
# plt.savefig('lnNvsM',bbox_inches='tight')
plt.show()
    
