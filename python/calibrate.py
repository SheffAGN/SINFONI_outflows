import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

def fluxcal(sci, std, ndit=5):
    
    #Perform flux calibration:
    #Get conversion from std:
    std.calcconv()

    #Convert to ctrt if not already done so:
    if (sci.flux.unit == u.ct / u.s):
        pass
    elif (sci.flux.unit == u.ct):
        exptime = sci.hdr['EXPTIME'] * ndit * u.s
        ctrt = sci.flux / exptime
    else:
        raise ValueError('Can only work with datacubes in cts or cts / s')

    conv = std.conv[:,np.newaxis,np.newaxis]
    sci.flux = conv * ctrt
    
    return sci

def telcorr(sci, std):
    
    #Don't need to worry about units here, since correction
    #is unitless:
    std.calctell()
    corr = std.tell[:,np.newaxis,np.newaxis]

    sci.flux = sci.flux / corr

    return sci