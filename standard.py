import numpy as np
from datacube import datacube
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.blackbody import blackbody_lambda as bb
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Box1DKernel


from scipy.integrate import simps

std = datacube()
std.read('Data/DataCubes/1327202/std/out_cube_obj00.fits')

#Calculate the Count Rate (need to check header exptime)
dit = std.hdr['EXPTIME']
ndit = 2.
ctrt = std.flux/(dit*ndit)

#Approximate star with blackbody:
bbwav = (std.lam * u.micron).to(u.AA)
bbtemp = 15200 * u.K
bbflux = bb(bbwav, bbtemp)*u.sr #units of ergs/s/cm2/A

#Interpolate K-band filter:
wav, resp = np.loadtxt('Ks.dat', usecols=(0, 1), unpack=True)
intresp = np.interp(bbwav.value, wav, resp, left=0., right=0.)

#Integrate over filter:
intbb = simps(bbflux*intresp, bbwav)/simps(intresp, bbwav)

#Normalise bbflux such that it matches the mag of the source:
kmag = 7.
flux0 = (4.283e-14 * u.W/(u.cm**2 * u.micron)).to(u.erg/(u.s*u.cm**2 * u.AA))
kflux = flux0*(10.**(-kmag/2.5))
norm = kflux/intbb
bbflux = norm*bbflux.value

#Extract the star:
#Find the maximum position:
#Collapse datacube, ignorin NaNs:
coll = np.nansum(std.flux, axis=0)

xmax, ymax = np.unravel_index(np.argmax(coll), coll.shape)
g_init = models.Gaussian2D(amplitude=np.max(coll), \
                           x_mean=xmax, y_mean=ymax, \
                           x_stddev=2.8, y_stddev=2.8)
        
fit_g = fitting.LevMarLSQFitter()
y, x = np.mgrid[:64, :64]
fit = fit_g(g_init, x, y, coll, weights=np.sqrt(np.abs(coll)))
mask = fit(x,y)/np.max(fit(x,y)) < 0.01 
mask = np.tile(mask, (bbwav.size, 1, 1))
masked = np.ma.array(data=ctrt, mask=mask)
ctrtspec = np.ma.sum(np.ma.sum(masked, axis=1), axis=1)
smooth = convolve(ctrtspec, Box1DKernel(10))


#print mask[1000:1200,33,17]
#print masked[1000:1200,33,17]
print ctrtspec[1000:1200].data
plt.plot(smooth)
plt.show()
quit()
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
