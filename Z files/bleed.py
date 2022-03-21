# -*- coding: utf-8 -*-
"""
Zoya - BLEEDING SOURCES AND NOISY BORDER

A piece of code that manually masks the borders and any sources that bleed
    These are caused by the CCD's unsuitable subexposure and other sources of noise due to 
    the geometry of the detector

I have created a dictionary of these bleeding sources as well as the coordinates for the borders, 
    these are all determined by eye from the DS9 visualiser. 

THE UPDATED BASE ARRAY IS SAVED AS A CSV FILE 'base_masked_bleeding.csv' SO WE DON'T HAVE TO RUN 
    THIS CODE AGAIN AND AGAIN'
"""
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

from astropy.io import fits

hdulist = fits.open(r'C:\Users\zoyaa\OneDrive\Documents\Year 3\Labs\Cycle 3\A1_mosaic.fits')
image = hdulist[0].data


mask = np.ones((4611,2570))
nx = 2570
ny = 4611


bg = 3416 - 300
bleed_circle = {
    1 : [[1430,3202],350],
    2 : [[906,2285],80],
    3 : [[976,2774],70],
    4 : [[2136,3756],60],
    5 : [[2464,3414],38],
    6 : [[2133,2308],40],
    7 : [[2089,1425],40],
    8 : [[2249,3298],100]
}

bleed_ellipse = {
    1 : [[778,3317],[90,100]],
    2 : [[2131,2311],[50,50]]}
    

bleed_box = {
    1 : [[1652,1106],[449,423]],
    2 : [[1703,1016],[343,310]],
    3 : [[1476,1395],[264,217]],
    4 : [[1541,1288],[158,111]],
        }

vertical_column = [1380,1600]
y_borders = [118,4517]
x_borders = [130,2440]
    

X = np.log(image[3050:3350,1230:1680])
Y = np.log(image[200:300,1400:1500])             
                

#%%  

"""
These functions parameterise the approximate shapes of the bleeding sources
                """
                
def circle(x,y):
    return x**2 + y**2 
    
def ellipse(x,y,a,b):
    return (x/a)**2 + (y/b)**2 



copy = np.copy(image) #Just for the purposes of visualisation 


"""
For loop goes through each of the `bleeding' positions in the dictonaries above 
    and masks these pixels in the base array. 
    The copy of the image array is also amended so we can visualise what's being 
    hidden.'
    
    The base array (which has masked values switched to 0) is then saved 
    externally as a csv file so we don't have to run this each time. 
    Saved under 'base_masked_bleeding.csv'
    
"""

for j in range (0,ny):
    for i in range (0,nx):
        
        if i < x_borders[0] or i > x_borders[1]:

            copy[j][i] = bg
            mask[j][i] = 0
            
        if j < y_borders[0] or j > y_borders[1]:
            copy[j][i] = bg
            mask[j][i] = 0
        
        for k in range (1,9):
            if circle(i-bleed_circle[k][0][0],j-bleed_circle[k][0][1]) < bleed_circle[k][1]**2: 
                copy[j][i] = bg
                mask[j][i] = 0
                
                
        for l in range(1,3):
            x = i-bleed_ellipse[l][0][0]
            y = j-bleed_ellipse[l][0][1]
            a = bleed_ellipse[l][1][0]
            b = bleed_ellipse[l][1][1]
            
            if ellipse(x,y,a,b) < 1: 
                copy[j][i] = bg
                mask[j][i] = 0
                
                
        for m in range (1,5):
            xmax = bleed_box[m][0][0]
            xmin = bleed_box[m][0][1]
            ymax = bleed_box[m][1][0]
            ymin = bleed_box[m][1][1]
            
            if xmin < i < xmax and ymin < j < ymax: 
                copy[j][i] = bg
                mask[j][i] = 0
        
        if vertical_column[0] < i < vertical_column[1]:
            copy[j][i] = bg
            mask[j][i] = 0
 

np.savetxt('base_masked_bleeding.cvs',mask,delimiter=",")
#%% THIS IS ALL PLOTTING 

"""
This cell plots a colormap of all the pixel values before and after the manual
    cuts. The colors on the map are all logarithmic values. 
"""

matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('axes',labelsize=30)

low = np.log(copy).min()
high = np.log(image).max()


plt.subplot(1,2,1)
plt.imshow(np.log(image),cmap='inferno',vmin=low,vmax=high)
plt.subplot(1,2,2)
plt.imshow(np.log(copy),cmap="inferno",vmin=low,vmax=high)
plt.show()

#%% THIS IS ALL PLOTTING 

# Z = np.log(copy[3050:3350,1230:1680])  

# plt.rcParams['font.family'] = 'serif'
# matplotlib.rc('xtick', labelsize=30) 
# matplotlib.rc('ytick', labelsize=30) 

# #matplotlib.rc('axes',labelsize=30)

# fig = plt.figure(figsize=(14.5, 20))
# fig.text(0.23, 0.06, r'Horizontal Pixel Index, - 1230', va='center', rotation='horizontal',fontsize=40)
# fig.text(0.1, 0.5, r'Vertical Pixel Index, - 3050', va='center', rotation='vertical',fontsize=40)

# ax1 = fig.add_subplot(211)
# im1 = ax1.imshow(np.log(image)[2000:4000,0:2400], cmap="inferno",vmin = Z.min() ,vmax = X.max())


# ax2 = fig.add_subplot(212)
# im2 = ax2.imshow(np.log(copy)[2000:4000,0:2400], cmap="inferno",vmin=Z.min(),vmax=X.max())


# #plt.setp([ax1,ax2], xticks=[], yticks=[])

# fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

# fig.colorbar(im1, cax=cbar_ax)
# fig.colorbar(im2, cax=cbar_ax)

# plt.savefig('remove_bleeding1.png',bbox_inches='tight')
# plt.show()