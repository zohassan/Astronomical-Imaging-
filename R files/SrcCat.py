#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:43:23 2022

@author: Rohith
"""

# %% Create Data & Remove backgrounds

from astropy.convolution import Gaussian2DKernel
from photutils.datasets import make_100gaussians_image
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.segmentation import deblend_sources

data = make_100gaussians_image()

bkg_estimator = MedianBackground()
bkg = Background2D(data, (50, 50), filter_size=(3, 3),bkg_estimator=bkg_estimator)
data -= bkg.background  # subtract the background
threshold = 2. * bkg.background_rms  # above the background

# %% Detect and Deblend sources

from astropy.stats import gaussian_fwhm_to_sigma
sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize()
npixels = 5
segm = detect_sources(data, threshold, npixels=npixels, kernel=kernel)
segm_deblend = deblend_sources(data, segm, npixels=npixels,kernel=kernel, nlevels=32,contrast=0.001)


# %% Source Catalogue 

from photutils.segmentation import SourceCatalog
cat = SourceCatalog(data, segm_deblend)
tbl = cat.to_table()
# tbl['xcentroid'].info.format = '.2f'  # optional format
# bl['ycentroid'].info.format = '.2f'
# tbl['kron_flux'].info.format = '.2f'
print(tbl)

# %% Plot of data with elliptical apertures

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
norm = simple_norm(data, 'sqrt')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Data')
cmap = segm_deblend.make_cmap(seed=123)
ax2.imshow(segm_deblend, origin='lower', cmap=cmap,interpolation='nearest')
ax2.set_title('Segmentation Image')
cat.plot_kron_apertures((2.5, 1.0), axes=ax1, color='white', lw=1.5)
cat.plot_kron_apertures((2.5, 1.0), axes=ax2, color='white', lw=1.5)


