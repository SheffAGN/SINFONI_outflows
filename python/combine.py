import matplotlib.pyplot as plt
import numpy as np
import glob
from datacube import science, standard
from astropy.stats import sigma_clip
from astropy.coordinates import Angle
from astropy import units as u
from calibrate import fluxcal, telcorr

#Get list of files:
scis = glob.glob('Data/DataCubes/1*/cor/*.fits')
stds = glob.glob('Data/DataCubes/1*/std/*.fits')

#B5V, B5V, B3V, B2V, B2V, B5V, B8V, B8V, B8V, B8V, B3V, DA2
mags = np.array([7.722,7.722,8.272,\
                 7.400,7.400,7.256,\
                 7.864,7.864,7.864,\
                 7.864,7.757,11.768])
temps = np.array([15200.,15200.,18800.,\
                  20800.,20800.,15200.,\
                  11400.,11400.,11400.,\
                  11400.,18800.,25200.])

#Read and align the first to get shape and normalisation:
sci = science()
sci.read(glob.glob('Data/DataCubes/*/cor/*.fits')[11])
sci.align()
std = standard(mag=mags[11],temp=temps[11])
std.read(stds[11])
sci = fluxcal(sci, std, ndit=5)
sci = telcorr(sci, std)
ind = np.logical_and(sci.lam.value>2.26, sci.lam.value<2.30)
ref = np.sum(sci.flux[ind,42,42])
rsci = sci

#Combine all datacubes into a single hypercube:
dims = np.shape(sci.flux)+np.shape(scis)
allsci = np.zeros(dims)

for i in range(len(scis)):
    
    sci = science()
    sci.read(scis[i])
    sci.align()

    std = standard(mag=mags[i],temp=temps[i])
    std.read(stds[i])

    sci = fluxcal(sci, std, ndit=5)
    sci = telcorr(sci, std)
    norm = ref / np.sum(sci.flux[ind,42,42])

    sci.flux = norm * sci.flux
    allsci[:,:,:,i] = sci.flux 
    
#Loop through the spaxels of the hypercube, taking the mean and stddev:
ysi = allsci.shape[1]
xsi = allsci.shape[2]
lsi = allsci.shape[0]

#Create "mean" and "stddev" datacubes:
msci = science()
ssci = science()
msci.lam = sci.lam
ssci.lam = sci.lam
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
 
wav = msci.lam
plt.plot(msci.lam, rsci.flux[:,42,42], 'r')
plt.plot(msci.lam, msci.flux[:,42,42], 'b')
plt.show()

#Save the output:
msci.write('mean.fits',header=sci.hdr)
ssci.write('stddev.fits',header=sci.hdr)
quit()

'''
ind = np.logical_and(dc.lam > 2.25, dc.lam < 2.35) 
spec = mdc.sflux[ind,42,42]
plt.subplot(7, 2, 14)
plt.plot(dc.lam[ind],spec,color='r')
plt.show()
quit()
'''