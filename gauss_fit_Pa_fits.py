from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting

def data_cube(file):

    z = 0.085

    hdulist = fits.open(file)

    hdulist.info()
    hdr = hdulist[0].header
    print(hdr)
    flux = hdulist[0].data
    print(flux.shape)

    lam = np.arange(float(hdr['CRVAL3'])-(float(hdr['NAXIS3'])/2)*float(hdr['CD3_3']), float(hdr['CRVAL3'])+(float(hdr['NAXIS3'])/2-0.5)*float(hdr['CD3_3']), float(hdr['CD3_3']))
    #print(len(lam))

    hdulist.close()


    return z, lam, flux



def gauss_paramters(file_param, x, y, lam):

    hdulist = fits.open(file_param)
    #hdulist.info()

    bestfit = hdulist[0].data
    #print(bestfit.shape)

    np.sum(bestfit, axis = 2)
    gauss_fit = bestfit[x,y,:]

    Pa_reg = (lam > 2.03536419463-0.02) & (lam < 2.03536419463+0.02)
    lam_Pa = lam[Pa_reg]
    #plt.plot(lam[Pa_reg],gauss_fit)
    #plt.show()


    flux_xy = np.sum(gauss_fit)



    return np.log(flux_xy)

def main():
    #z, lam, flux = data_cube('/Users/charlotteavery/Documents/SURE project/COADD_mean3sig.fits')

    #flux_img = np.zeros((67,64))

    #for x in range(0,67):
    #    for y in range(0,64):

    #        flux_xy = gauss_paramters('/Users/charlotteavery/Documents/SURE project/Pa_alpha_gaussfit_trial_sn.fits', x, y, lam)
    #        flux_img[x,y] = flux_xy

    hdulist=fits.open('Pa_alpha_Flux.fits')
    flux_img=hdulist[0].data

    plt.imshow(flux_img, vmax=0.02*np.max(flux_img), origin = 'lower')

    cb = plt.colorbar()
    cb.set_label('log(flux)')

    plt.show()
main()
