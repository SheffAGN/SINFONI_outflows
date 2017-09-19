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
                                   x_stddev=2.8)
        y, x = np.mgrid[:64, :64]
        fit_g = fitting.LevMarLSQFitter()
        fit = fit_g(g_init, x, y, coll, weights=np.sqrt(np.abs(coll)))
        
        #Calculate shift:
        xshift = int(32-np.round(fit.x_mean.value))
        yshift = int(32-np.round(fit.y_mean.value))

        #Create a larger 3D array to put shifted cube into:
        flux = np.zeros((np.shape(self.flux)[0],84,84))
        flux[:] = np.NaN
        xstart = 42-32+xshift
        ystart = 42-32+yshift
        flux[:,ystart:ystart+64,xstart:xstart+64] = self.flux

        #Set the aligned flag to True    
        self.flux = flux * self.flux.unit
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

    def __init__(self, mag=7, temp=15200.):
        super().__init__()

        self.mag = mag
        self.temp = temp

        #Bookkeeping:
        self.extracted = False
        self.fitted = False
        self.calibrated = False
    
    def genbb(self):

        #Generate BB with given temp:
        return bb(self.lam.to(u.AA), self.temp * u.K) * u.sr

    def genfilter(self, fname='Ks.dat', \
                  zeromag=4.283e-14 * u.W/(u.cm**2 * u.micron)):

        #Interpolates filter onto wav array:
        wav, resp = np.loadtxt(fname, usecols=(0, 1), unpack=True)
        filtresp = np.interp(self.lam.to(u.AA).value, wav, resp,\
                                  left=0., right=0.)

        return filtresp, zeromag 

    def calbb(self):
    
        #Calibrate the bb so it matches input mag:
        #Integrate over filter response function:
        bbflux = self.genbb()
        filtresp, zmag = self.genfilter()
        intbb = simps(bbflux * filtresp, self.lam.to(u.AA))/\
                simps(filtresp, self.lam.to(u.AA))
        
        #Normalise bbflux such that it matches the mag of the source:
        flux0 = zmag.to(u.erg/(u.s * u.cm**2 * u.AA))
        flux = flux0 * (10.**(-self.mag/2.5))
        norm = flux/intbb

        self.calibrated = True
        self.cal = norm * bbflux.value

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
        ctspec = np.ma.filled(np.ma.sum(np.ma.sum(masked, axis=1), axis=1),0.)
        ctspec[np.isnan(ctspec)] = 0.
        
        return ctspec

    def getctrt(self, ndit=2):
        #Convert into countrate:
        exptime = self.hdr['EXPTIME'] * ndit * u.s

        self.extracted = True
        self.ctrt = self.extract() / exptime

    def getpts(self):

        #Smooth the spectrum:
        if ~self.extracted:
            self.getctrt()
        
        smooth = convolve(self.ctrt.value, Box1DKernel(10)) * u.ct / u.s

        #Select wavelengths devoid of tellurics
        selwav = np.array([[1.982,1.985],[2.034,2.039],[2.09,2.100],\
                           [2.140,2.150],[2.230,2.240],[2.31,2.315],\
                           [2.330,2.340],[2.360,2.365],[2.398,2.402],\
                           [2.438,2.440]])
        means = np.zeros_like(selwav)

        i = 0
        for row in selwav:
            mask = ~np.logical_and(self.lam.to(u.micron).value>=row[0],\
                                   self.lam.to(u.micron).value<=row[1])
            means[i,0] = np.ma.mean(np.ma.array(data=self.lam, mask=mask)).value
            means[i,1] = np.ma.mean(np.ma.array(data=smooth, mask=mask)).value
            i += 1

        return means

    def fitpts(self):
        
        #Fit means with a polynomial:
        means = self.getpts()
        p_init = models.Polynomial1D(degree=4)
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, means[:,0], means[:,1])
        
        self.fitted = True
        self.fit = p(self.lam.to(u.micron).value) * u.ct / u.s

    def calcconv(self):

        if ~self.fitted:
            self.fitpts()
        if ~self.calibrated:
            self.calbb()

        self.conv = self.cal / self.fit

    def calctell(self):
        
        if ~self.extracted:
            self.getctrt()

        if ~self.fitted:
            self.fitpts()       

        tell = self.ctrt / self.fit
        
        ind  = np.logical_or(tell == 0, np.isnan(tell)) 
        tell[ind] = 0.
        tell = tell / np.max(tell)
        tell[ind] = 1.
        
        self.tell = tell
