import numpy as np
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, ImageNormalize
from datacube import datacube
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

#Read in the mean and stddev datacubes:
msci = datacube()
msci.read('mean.fits')
#sdc = datacube()
#sdc.read('stddev.fits')

#Stack
image = msci.flux[353,:,:]
norm = ImageNormalize(image, interval=ZScaleInterval())

plt.imshow(msci.flux[353,:,:], origin='lower', norm=norm)
plt.show()

sub = np.nanmean(np.nanmean(msci.flux[:,40:44,40:44], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
plt.plot(msci.lam, smooth, 'b')

sub = np.nanmean(np.nanmean(msci.flux[:,42:49,25:31], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
plt.plot(msci.lam, smooth, 'r')

sub = np.nanmean(np.nanmean(msci.flux[:,63:72,35:48], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
plt.plot(msci.lam, smooth, 'g')

plt.show() 


