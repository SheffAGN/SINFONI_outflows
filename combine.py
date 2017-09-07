import matplotlib.pyplot as plt
import numpy as np
import glob
from datacube import datacube

#Get list of files:
filenames = glob.glob('Data/DataCubes/*/cor/*.fits')

#Read and align the first to get shape:
dc = datacube()
dc.read(filenames[0])
dc.align()

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
    
#Loop through the spaxels of the hypercube, taking the mean and stddev:
ysi = hcube.shape[1]
xsi = hcube.shape[2]
lsi = hcube.shape[0]

#Create "mean" and "stddev" datacubes:
mdc = datacube()
sdc = datacube()
mdc.sflux = np.full((lsi, ysi, xsi), np.NaN)
sdc.sflux = np.full((lsi, ysi, xsi), np.NaN)

#Loop through spaxels, taking average while ignoring NaNs: 
for xpos in range(xsi):
    for ypos in range(ysi):
        spax = hcube[:,ypos,xpos,:]
        mdc.sflux[:,ypos,xpos] = np.nanmedian(spax, axis=1)
        sdc.sflux[:,ypos,xpos] = np.nanmedian(spax, axis=1)
        
#Save the output:
mdc.save('mean.fits',header=dc.hdr)
sdc.save('stddev.fits',header=dc.hdr)
