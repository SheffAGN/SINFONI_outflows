import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from astropy.io import fits
from astropy import units as u
from astropy.modeling import models, fitting
from astropy.modeling.blackbody import blackbody_lambda as bb
from astropy.convolution import convolve, Box1DKernel

class datacube():

    def __init__(self):
        self.flux = np.array([])

    def read(self, fname):
        hdulist = fits.open(fname)

        self.hdr = hdulist[0].header
        self.flux = hdulist[0].data * u.ct

        wavstart = float(self.hdr['CRVAL3'])-(float(self.hdr['NAXIS3'])/2)*float(self.hdr['CDELT3']) 
        wavend = float(self.hdr['CRVAL3'])+(float(self.hdr['NAXIS3'])/2-0.5)*float(self.hdr['CDELT3'])
        dwav = float(self.hdr['CDELT3'])
        self.lam = np.arange(wavstart, wavend, dwav) * u.micron
    
        hdulist.close()

    def write(self,fname,header=None):
        fits.writeto(fname,self.sflux,header=header,overwrite=True)

class science(datacube):

    def __init__(self):
        
        super().__init__()
        self.aligned = False
        self.normalised = False
        self.sflux = np.array([])

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
        spec = self.sflux[20:2000,:,:]
        plt.figure(figsize=(8, 2.5))
        plt.subplot(2, 7, 1)
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
        ind = np.logical_and(self.lam > 2.0, self.lam < 2.35)
        #self.coll = np.nansum(self.sflux[ind,37:47,37:47])
        self.coll = np.nansum(self.sflux[ind,42,42])
        norm = 1e7/self.coll
        self.sflux = norm*self.sflux

        #Set the normalised flag to True    
        self.normalised = True

class standard(datacube):

    def __init__(self):
        super().__init__()
        self.bbcal = False
        self.fluxcal = False

    def genbb(self, temp=15200):
        bbwav = self.lam.to(u.AA)
        self.bbflux = bb(bbwav, temp * u.K) * u.sr

    def genfilter(self, fname='Ks.dat', zeromag=4.283e-14):
        #Interpolates filter onto wav array:
        wav, resp = np.loadtxt(fname, usecols=(0, 1), unpack=True)
        self.filtresp = np.interp(self.lam.to(u.AA).value, wav, resp,\
                                  left=0., right=0.)
        self.zmag = zeromag

    def calbb(self, mag=7.):
    
        #Calibrate the bb so it matches input mag:
        #Integrate over filter response function:
        intbb = simps(self.bbflux * self.filtresp, self.lam.to(u.AA))/\
                simps(self.filtresp, self.lam.to(u.AA))
        
        #Normalise bbflux such that it matches the mag of the source:
        flux0 = (self.zmag * u.W/(u.cm**2 * u.micron)).\
                to(u.erg/(u.s * u.cm**2 * u.AA))
        flux = flux0 * (10.**(-mag/2.5))
        norm = flux/intbb
        self.bbflux = norm*self.bbflux.value
        self.bbcal = True

    def extract(self, ndit=2):
        #Extract the star:
        #First, collapse along wav dimension:
        coll = np.nansum(self.flux, axis=0)

        #Find maximum position as starting point to fit:
        xmax, ymax = np.unravel_index(np.argmax(coll), coll.shape)
        g_init = models.Gaussian2D(amplitude=np.max(coll), \
                                   x_mean=xmax, y_mean=ymax, \
                                   x_stddev=2.8, y_stddev=2.8)

        #Perform 2D Gaussian fit:
        fit_g = fitting.LevMarLSQFitter()
        y, x = np.mgrid[:64, :64]
        fit = fit_g(g_init, x, y, coll, weights=np.sqrt(np.abs(coll)))

        #Create mask from fit:
        mask = fit(x,y)/np.max(fit(x,y)) < 0.01 
        mask = np.tile(mask, (self.lam.size, 1, 1))
        masked = np.ma.array(data=self.flux, mask=mask)
        
        #Sum unmasked pixels to extract:
        self.ctspec = np.ma.sum(np.ma.sum(masked, axis=1), axis=1)

    def getctrt(self, ndit=2):
        #Convert into countrate:
        exptime = self.hdr['EXPTIME'] * ndit * u.s
        smooth = convolve(self.ctspec, Box1DKernel(10)) * u.ct

        self.ctspec = smooth / exptime

    def getpts(self):

        selwav = np.array([[1.982,1.985],[2.034,2.039],[2.09,2.1],\
                  [2.14,2.15],[2.23,2.24],[2.31,2.315],[2.33,2.34],\
                  [2.36,2.365],[2.398,2.402],[2.438,2.44]])
        self.means = np.zeros_like(selwav)

        i = 0
        for row in selwav:
            mask = ~np.logical_and(self.lam.to(u.micron).value>=row[0],\
                                   self.lam.to(u.micron).value<=row[1])
            self.means[i,0] = np.ma.mean(np.ma.array(data=self.lam,mask=mask)).value
            self.means[i,1] = np.ma.mean(np.ma.array(data=self.ctspec,mask=mask)).value
            i += 1

    def fitpts(self):
        
        #Fit means with a polynomial:
        p_init = models.Polynomial1D(degree=4)
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, self.means[:,0], self.means[:,1])
        self.ctspecfit = p(self.lam.to(u.micron).value) * u.ct / u.s

    def getconv(self):

        self.conv = self.bbflux / self.ctspecfit

        plt.plot(self.lam, self.conv * self.ctspec)
        plt.plot(self.lam, self.bbflux)
        plt.show()