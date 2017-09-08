import numpy as np
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, ImageNormalize
from datacube import datacube
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel

#Read in the mean and stddev datacubes:
mdc = datacube()
mdc.read('mean.fits')
#sdc = datacube()
#sdc.read('stddev.fits')

#Stack
#image = mdc.flux[441,:,:]
#norm = ImageNormalize(image, interval=ZScaleInterval())

#plt.imshow(mdc.flux[440,:,:], origin='lower', norm=norm)
#plt.show()

sub = np.nanmean(np.nanmean(mdc.flux[:,40:44,40:44], axis=1), axis=1)
smooth = convolve(sub, Box1DKernel(6))
plt.plot(mdc.lam, smooth)
plt.show() 


