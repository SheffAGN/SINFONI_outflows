from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import glob


class datacube():

    def __init__(self):
        self.aligned = False

    def read(self, fname):
        hdulist = fits.open(fname)

        hdr = hdulist[0].header
        self.flux = hdulist[0].data

        wavstart = float(hdr['CRVAL3'])-(float(hdr['NAXIS3'])/2)*float(hdr['CDELT3']) 
        wavend = float(hdr['CRVAL3'])+(float(hdr['NAXIS3'])/2-0.5)*float(hdr['CDELT3'])
        dwav = float(hdr['CDELT3'])
        self.lam = np.arange(wavstart, wavend, dwav)
    
        hdulist.close()

    def align(self):

        #Collapse datacube, ignorin NaNs:
        coll = np.nansum(self.flux, axis=0)
        
        #Fit a 2D Gaussian to the collapsed cube:
        xmax, ymax = np.unravel_index(np.argmax(coll), coll.shape)
        g_init = models.Gaussian2D(amplitude=np.max(coll), x_mean=xmax, y_mean=ymax, x_stddev=2.8, y_stddev=2.8)

        y, x = np.mgrid[:64, :64]
        
        np.sqrt(np.abs(coll))

        fit_g = fitting.LevMarLSQFitter()
        fit = fit_g(g_init, x, y, coll, weights=np.sqrt(np.abs(coll)))
        print fit
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 2, 1)
        plt.imshow(coll)
        plt.title("Data")
        plt.subplot(1, 2, 2)
        plt.imshow(fit(x, y))
        plt.show()

#Get list of files:
files = glob.glob('Data/DataCubes/*/cor/*.fits')

#Need to positionally align each cube with each other. 

#Read one file to get shape:
dc1 = datacube()
dc1.read(files[0])
dc1.align()

#Combine all datacubes into a single hypercube:
'''
dims = np.shape(flux)+np.shape(files)
hcube = np.zeros(dims)

i = 0
for file in files:
    lam, flux = data(file)
    hcube[:,:,:,i] = flux 
    i += 1
    
#Loop through the 
'''