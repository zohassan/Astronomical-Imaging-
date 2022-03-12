# -*- coding: utf-8 -*-
"""
STATISTICS OF THE IMAGE - FORMING A HISTOGRAM 
"""
import numpy as np
from scipy.optimize import curve_fit 
from scipy.stats import norm 
import matplotlib.pyplot as plt 
from astropy.io import fits


hdulist = fits.open(r'C:\Users\zoyaa\OneDrive\Documents\Year 3\Labs\Cycle 3\A1_mosaic.fits')
image = hdulist[0].data


# The base array is in the form of 1s and 0s, changed that to True and False
base = np.genfromtxt('base_masked_bleeding.csv',delimiter=',')
base = base > 0

nx = 2570
ny = 4611


"""
Counts is a flattened array that include all the same elements of image which 
    correspond to `True' in the base array'
    """
counts = image[base]
        
#%%   PLOTTING A HISTOGRAM         
            
def gaussian(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

n,bins,patches = plt.hist(counts,100000,histtype='stepfilled',alpha=1,color='green',density=True)
xdata = np.delete(bins,-1)
fit,pcov = curve_fit(gaussian,xdata,n,p0=[100,3500,200])
xarray = np.linspace(3000,5000,1000)


plt.plot(xarray,gaussian(xarray,2.8*fit[0],fit[1],fit[2]),'r--',lw= 4)
plt.annotate('$\mu$ = 3418.33$\pm$0.07',(3450,0.07),fontsize = 35)
plt.annotate('$\sigma$ = 11.90$\pm$0.07', (3450,0.06), fontsize=35)


#BOILER CODE for plotting 
plt.xlabel('Photons Per Pixel',fontsize = 40)
plt.ylabel('Normalised Count',fontsize = 40)
#plt.legend(fontsize = 35)
#plt.title('Ge,273K, 0.02 - 0.15 V',fontsize = 33)
plt.xlim(3350,3500)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (20,10)
plt.grid(True)
plt.tick_params(axis = 'both',labelsize = 33)
#plt.savefig('counts_distribution_526',bbox_inches='tight')
plt.show()