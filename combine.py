import matplotlib.pyplot as plt
import numpy as np
import glob
from datacube import datacube
from astropy.stats import sigma_clip

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
plt.figure(figsize=(15, 10))
for filename in filenames:
    dc = datacube()
    dc.read(filename)
    dc.align()

    #Don't bother normalising, it doesn't help
    #(should also not be needed when flux calibrated)
    #dc.normalise()

    hcube[:,:,:,i] = dc.sflux 
    i += 1
    
    ind = np.logical_and(dc.lam > 2.0, dc.lam < 2.35) 
    spec = dc.sflux[ind,40,42]
    wav = dc.lam[ind]
    #plt.subplot(7, 2, i)
    #plt.plot(wav, spec)



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
        allnan = np.all(np.isnan(spax))
        if allnan == False:
            clipped = sigma_clip(spax, axis=1)
            mdc.sflux[:,ypos,xpos] = np.ma.mean(clipped, axis=1)
            sdc.sflux[:,ypos,xpos] = np.ma.std(clipped, axis=1)

ind = np.logical_and(dc.lam > 2.0, dc.lam < 2.35) 
spec = mdc.sflux[ind,22,22]
wav = dc.lam[ind]
plt.plot(wav, spec)
plt.show()

#Save the output:
mdc.save('mean.fits',header=dc.hdr)
sdc.save('stddev.fits',header=dc.hdr)

'''
ind = np.logical_and(dc.lam > 2.25, dc.lam < 2.35) 
spec = mdc.sflux[ind,42,42]
plt.subplot(7, 2, 14)
plt.plot(dc.lam[ind],spec,color='r')
plt.show()
quit()
'''