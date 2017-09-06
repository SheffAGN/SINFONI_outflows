from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import glob

class datacube():

    def __init__(self):
        self.aligned = False
        self.normalised = False
        self.flux = np.array([])
        self.sflux = np.array([])

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

    

#Get list of files:
filenames = glob.glob('Data/DataCubes/*/cor/*.fits')

#Read and align the first to get shape:
dc = datacube()
dc.read(filenames[0])
dc.align()
dc.normalise()

#Combine all datacubes into a single hypercube:
dims = np.shape(dc.sflux)+np.shape(filenames)
hcube = np.zeros(dims)

i = 0
for filename in filenames:
    dc = datacube()
    dc.read(filename)
    dc.align()
    dc.normalise()

    hcube[:,:,:,i] = dc.sflux 
    i += 1
    
#Loop through the spaxels of the datacube, taking the mean and stddev:
ysi = hcube.shape[1]
xsi = hcube.shape[2]
lsi = hcube.shape[0]

#Create "mean" and "stddev" datacubes:
mdc = datacube()
sdc = datacube()
mdc.sflux = np.full((lsi, ysi, xsi), np.NaN)
sdc.sflux = np.full((lsi, ysi, xsi), np.NaN)

#Ignore NaNs when taking mean 
for xpos in range(xsi):
    for ypos in range(ysi):
        spax = hcube[:,ypos,xpos,:]
        mdc.sflux[:,ypos,xpos] = np.nanmean(spax, axis=1)
        sdc.sflux[:,ypos,xpos] = np.nanstd(spax, axis=1)
        
coll = np.nansum(mdc.sflux, axis=0)
sdc.sflux = sdc.sflux/np.sqrt(13.)
print mdc.sflux[1000:1010,42,42]
print sdc.sflux[1000:1010,42,42]
plt.plot(dc.lam, mdc.sflux[:,22,22])
plt.show()

#Next: Save the mean and stddev datacubes in their own fits files
#Use the header from one of the original datacubes. 
