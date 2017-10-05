import numpy as np
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, ImageNormalize
from datacube import datacube, standard
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

#Read in the mean and stddev datacubes:
msci = datacube()
msci.read('../Data/DataCubes/mean.fits')
#sdc = datacube()
#sdc.read('stddev.fits')

std = standard(mag=7.722,temp=15200)
std.read('../Data/DataCubes/1327202/std/out_cube_obj00.fits')
std.calctell()

#Stack
image = msci.flux[353,:,:]
norm = ImageNormalize(image, interval=ZScaleInterval())

plt.imshow(msci.flux[353,:,:], origin='lower', norm=norm)
plt.plot([21.5,21.5,32.5,32.5,21.5],[42.5,51.5,51.5,42.5,42.5])
plt.plot([50.5,50.5,61.5,61.5,50.5],[29.5,38.5,38.5,29.5,29.5])
plt.show()

sub = np.nanmean(np.nanmean(msci.flux[:,40:44,40:44], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
plt.plot(msci.lam, smooth, 'b')

sub = np.nanmean(np.nanmean(msci.flux[:,43:51,22:32], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
plt.plot(msci.lam, 50*smooth, 'r')
#plt.plot(std.lam, 1e-19*std.tell)

sub = np.nanmean(np.nanmean(msci.flux[:,30:38,51:61], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
#plt.plot(msci.lam, smooth, 'g')

plt.show() 


