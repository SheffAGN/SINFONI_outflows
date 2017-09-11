import numpy as np
from datacube import datacube
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.modeling.blackbody import blackbody_nu

std = datacube()
std.read('Data/DataCubes/1327202/std/out_cube_obj00.fits')

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

