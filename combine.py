import matplotlib.pyplot as plt
import numpy as np
import glob
from datacube import science, standard
from astropy.stats import sigma_clip

#Get list of files:
filenames = glob.glob('Data/DataCubes/*/cor/*.fits')

#Read and align the first to get shape:
sci = science()
sci.read(filenames[0])
sci.align()

std = standard()
std.read('Data/DataCubes/1327202/std/out_cube_obj00.fits')
std.process()

std.conv = std.conv[:,np.newaxis, np.newaxis]
std.tell = std.tell[:,np.newaxis, np.newaxis]

xx = sci.sflux*std.conv
plt.plot(sci.lam, xx[:,42,42])
xx = xx/std.tell
plt.plot(sci.lam, xx[:,42,42])
plt.show()
quit()
#Combine all datacubes into a single hypercube:
dims = np.shape(sci.sflux)+np.shape(filenames)
allsci = np.zeros(dims)

i = 0
plt.figure(figsize=(15, 10))
for filename in filenames:
    sci = datacube()
    sci.read(filename)
    sci.align()

    #Don't bother normalising, it doesn't help
    #(should also not be needed when flux calibrated)
    #dc.normalise()

    allsci[:,:,:,i] = sci.sflux 
    i += 1
    
#Loop through the spaxels of the hypercube, taking the mean and stddev:
ysi = allsci.shape[1]
xsi = allsci.shape[2]
lsi = allsci.shape[0]

#Create "mean" and "stddev" datacubes:
msci = datacube()
ssci = datacube()
msci.flux = np.full((lsi, ysi, xsi), np.NaN)
ssci.flux = np.full((lsi, ysi, xsi), np.NaN)

#Loop through spaxels, taking average while ignoring NaNs: 
for xpos in range(xsi):
    for ypos in range(ysi):
        spax = allsci[:,ypos,xpos,:]
        allnan = np.all(np.isnan(spax))
        if not allnan:
            clipped = sigma_clip(spax, axis=1)
            msci.flux[:,ypos,xpos] = np.ma.mean(clipped, axis=1)
            ssci.flux[:,ypos,xpos] = np.ma.std(clipped, axis=1)

ind = np.logical_and(msci.lam > 2.0, msci.lam < 2.35) 
spec = mdc.sflux[ind,22,22] 
wav = dc.lam[ind]
plt.plot(wav, spec)
plt.show()
quit()
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