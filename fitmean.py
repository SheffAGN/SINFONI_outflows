import numpy as np
from astropy.modeling import models, fitting
from datacube import datacube
import matplotlib.pyplot as plt

#Read in the mean and stddev datacubes:
mdc = datacube()
mdc.read('mean.fits')
#sdc = datacube()
#sdc.read('stddev.fits')

#Stack
plt.imshow(np.log10(mdc.flux[441,:,:]))
plt.show()

