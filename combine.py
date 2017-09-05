from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import glob

def data(file):

    hdulist = fits.open(file)

    hdr = hdulist[0].header
    flux = hdulist[0].data

    wavstart = float(hdr['CRVAL3'])-(float(hdr['NAXIS3'])/2)*float(hdr['CDELT3']) 
    wavend = float(hdr['CRVAL3'])+(float(hdr['NAXIS3'])/2-0.5)*float(hdr['CDELT3'])
    dwav = float(hdr['CDELT3'])
    lam = np.arange(wavstart, wavend, dwav)
    
    hdulist.close()

    return lam, flux

#Get list of files:
files = glob.glob('Data/DataCubes/*/cor/*.fits')

#Need to positionally align each cube with each other. 

#Read one file to get shape:
lam0, flux = data(files[0])

#Combine all datacubes into a single hypercube:
dims = np.shape(flux)+np.shape(files)
hcube = np.zeros(dims)

i = 0
for file in files:
    lam, flux = data(file)
    hcube[:,:,:,i] = flux 
    i += 1
    
#Loop through the 