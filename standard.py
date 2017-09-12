import numpy as np
from datacube import datacube
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.blackbody import blackbody_lambda as bb
from scipy.integrate import simps

std = datacube()
std.read('Data/DataCubes/1327202/std/out_cube_obj00.fits')

#Calculate the Count Rate (need to check header exptime)
exptime = std.hdr['EXPTIME']
ctrt = std.flux/exptime

#Approximate star with blackbody:
bbwav = (std.lam * u.micron).to(u.AA)
bbtemp = 15200 * u.K
bbflux = bb(bbwav, bbtemp)*u.sr #units of ergs/s/cm2/A

#Interpolate K-band filter:
wav, resp = np.loadtxt('Ks.dat', usecols=(0, 1), unpack=True)
intresp = np.interp(bbwav.value, wav, resp, left=0., right=0.)

#Integrate over filter to get Ks mag of BB:
mag = 7.
flux0 = (4.283e-14 * u.W/(u.cm**2 * u.micron)).to(u.erg/(u.s*u.cm**2 * u.AA))
kflux = flux0*(10.**(-mag/2.5))
intbb = simps(bbflux*intresp, bbwav)/simps(intresp, bbwav)
norm = kflux/intbb
bbflux = norm*bbflux.value
print bbflux
quit()

print bbflux
quit()
plt.plot(bbwav.value, bbflux)
plt.show()
quit()

#Extract the star:
ypos, xpos = np.mgrid[0:64,0:64]
print xpos

'''
agn = datacube()
agn.read('Data/DataCubes/1327202/cor/out_cube_obj_cor00.fits')

bbwav = std.lam * u.micron
bbtemp = 15200 * u.K
bbflux = blackbody_nu(bbwav, bbtemp)

norm = 13000./0.000070
plt.plot(bbwav.value, norm*bbflux.value, color='g')

coll = np.nansum(np.nansum(std.flux[:,30:34,15:19], axis=1), axis=1)

#plt.subplot(2, 1, 1)
plt.plot(std.lam, coll, color='r')

coll = np.nansum(np.nansum(agn.flux[:,30:34,30:34], axis=1), axis=1)
#plt.subplot(2, 1, 2)
plt.plot(agn.lam, 12*coll, color='b')

plt.show()
'''
