from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting

class datacube():

    def __init__(self):
        self.aligned = False
        self.normalised = False
        self.flux = np.array([])
        self.sflux = np.array([])

    def read(self, fname):
        hdulist = fits.open(fname)

        self.hdr = hdulist[0].header
        self.flux = hdulist[0].data

        wavstart = float(self.hdr['CRVAL3'])-(float(self.hdr['NAXIS3'])/2)*float(self.hdr['CDELT3']) 
        wavend = float(self.hdr['CRVAL3'])+(float(self.hdr['NAXIS3'])/2-0.5)*float(self.hdr['CDELT3'])
        dwav = float(self.hdr['CDELT3'])
        self.lam = np.arange(wavstart, wavend, dwav)
    
        hdulist.close()

    def align(self):

        #Collapse datacube, ignorin NaNs:
        coll = np.nansum(self.flux, axis=0)
        
        #Fit a 2D Gaussian to the collapsed cube:
        xmax, ymax = np.unravel_index(np.argmax(coll), coll.shape)
        g_init = models.Gaussian2D(amplitude=np.max(coll), \
                                   x_mean=xmax, y_mean=ymax, \
                                   x_stddev=2.8, y_stddev=2.8)
        y, x = np.mgrid[:64, :64]
        fit_g = fitting.LevMarLSQFitter()
        fit = fit_g(g_init, x, y, coll, weights=np.sqrt(np.abs(coll)))
        
        #Calculate shift:
        xshift = int(32-np.round(fit.x_mean.value))
        yshift = int(32-np.round(fit.y_mean.value))

        #Create a larger 3D array to put shifted cube into:
        self.sflux = np.zeros((np.shape(self.flux)[0],84,84))
        self.sflux[:] = np.NaN
        xstart = 42-32+xshift
        ystart = 42-32+yshift
        self.sflux[:,ystart:ystart+64,xstart:xstart+64] = self.flux

        #Set the aligned flag to True    
        self.aligned = True
        
        '''
        coll2 = np.nansum(self.sflux, axis=0)
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 3, 1)
        plt.imshow(coll)
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(fit(x, y))
        plt.subplot(1, 3, 3)
        plt.imshow(coll2)
        plt.plot([42], [42], '+', color='r')
        plt.show()
        '''

    def normalise(self):

        #Normalise the cube so that the central region sums to (an arbitrary) 10^7
        coll = np.nansum(self.sflux[:,37:47,37:47])
        norm = 1e7/coll
        self.sflux = norm*self.sflux

        #Set the normalised flag to True    
        self.normalised = True

    def save(self,fname,header=None):

        fits.writeto(fname,self.sflux,header=header,overwrite=True)
