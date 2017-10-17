import numpy as np
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.io import fits
from datacube import datacube, standard
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel
from scipy.stats import chi2

#Read in the mean and stddev datacubes:
msci = datacube()
msci.read('../Data/DataCubes/mean.fits')
sdc = datacube()
sdc.read('../Data/DataCubes/stddev.fits')

#Based on initial fitting results, errors are overestimated by
#a factor of 2.5:
sdc.flux = sdc.flux / 2.5

#Redshift the spectra:
z = 0.085
msci.lam = msci.lam / (1. + z)
sdc.lam = sdc.lam / (1. + z)

#Rest wav of H2:
h2rw = 2.1218

#Extract the region around H2:
o = np.where(np.logical_and(msci.lam.value >= h2rw - 0.03, \
                            msci.lam.value < h2rw + 0.03))

#If you want to plot error bars:
#plt.errorbar(msci.lam[o].value, \
#             np.squeeze(msci.flux[o,y,x].value), \
#             yerr=np.squeeze(sdc.flux[o,y,x].value))

#Loop over the spaxels, fitting a Gaussian for H2 where necessary:
#Define fitting algorithm:
fitAlg = fitting.LevMarLSQFitter()
velShift = np.zeros([84,84])-1000.

for x in np.arange(0,83):
    for y in np.arange(0,83):

        #Perform the fit:
        # Fit continuum using a polynomial:
        cInit = models.Polynomial1D(3)
        cFit = fitAlg(cInit, \
                      msci.lam[o].value, \
                      np.squeeze(msci.flux[o,y,x].value))
        dModel = np.squeeze(msci.flux[o,y,x].value) - cFit(msci.lam[o].value)
        cChisq = np.sum(dModel**2 / np.squeeze(sdc.flux[o,y,x].value)**2)

        #Fit the line by *adding* a Gaussian to the fitted continuum:
        lInit = models.Gaussian1D(np.max(msci.flux[o,y,x].value),
                                   2.1218, 0.005, fixed={'mean':True})
        clInit = cFit + lInit
        clFit = fitAlg(clInit, \
                      msci.lam[o].value, \
                      np.squeeze(msci.flux[o,y,x].value), 
                      weights=1./np.squeeze(sdc.flux[o,y,x].value))
        dModel = np.squeeze(msci.flux[o,y,x].value) - clFit(msci.lam[o].value)
        clChisq = np.sum(dModel**2 / np.squeeze(sdc.flux[o,y,x].value)**2)

        #Calculate change in chisq:
        dChisq = cChisq - clChisq

        #If it is significant, allow mean to change:
        if np.isfinite(dChisq) and dChisq > chi2.ppf(0.997,3):
            clFit.mean_1.fixed = False
            clFit2 = fitAlg(clFit, \
                      msci.lam[o].value, \
                      np.squeeze(msci.flux[o,y,x].value), 
                      weights=1./np.squeeze(sdc.flux[o,y,x].value))
            dModel = np.squeeze(msci.flux[o,y,x].value) - \
                     clFit2(msci.lam[o].value)
            clChisq2 = np.sum(dModel**2 / np.squeeze(sdc.flux[o,y,x].value)**2)
            dChisq = clChisq - clChisq2
            
            #Does varying the mean help?
            if dChisq > chi2.ppf(0.997,1):
                velShift[y,x] = 3e5*(clFit2.mean_1-2.1218)/2.1218      
            else:
                velShift[y,x] = 0.

        print(x,y)

fits.writeto('velShift.fits', velShift, overwrite=True)

#norm = ImageNormalize(velShift, interval=ZScaleInterval())
plt.imshow(velShift, origin='lower')
plt.show()

#Fit the continuum:
#lc = np.logical_and(msci.lam.value >= h2rw - 0.03, \
#                    msci.lam.value < h2rw - 0.007)
#uc = np.logical_and(msci.lam.value >= h2rw + 0.007, \
#                    msci.lam.value < h2rw + 0.03)
#oc = np.where(np.logical_or(lc, uc))
