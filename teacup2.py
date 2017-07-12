from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting

def data(file):

    z = 0.085

    hdulist = fits.open(file)

    hdulist.info()
    hdr = hdulist[0].header
    #print(hdr)
    flux = hdulist[0].data
    print(flux.shape)

    lam = np.arange(float(hdr['CRVAL3'])-(float(hdr['NAXIS3'])/2)*float(hdr['CD3_3']), float(hdr['CRVAL3'])+(float(hdr['NAXIS3'])/2-0.5)*float(hdr['CD3_3']), float(hdr['CD3_3']))
    #print(len(lam))

    hdulist.close()

    return z, lam, flux

#--------------------------------------------------------------------------------------------------------------------------


def spaxel_fit(inlam, influx, plot=False):

    if np.all(influx==0) == True:
        print('zeros')
        return 0.

    o = ~np.isnan(influx)
    lam = inlam[o]
    spaxel_flux = influx[o]

    #remove outliers
    #spaxel_flux[np.where(spaxel_flux > 4e-15)] = np.nan
    #spaxel_flux[np.where(spaxel_flux < -1e-15)] = np.nan
    #spaxel_flux[np.where(spaxel_flux == 0)] = np.nan


    big_flux = spaxel_flux*(1e16)
    big_noise = np.sqrt(big_flux)
    spaxel_noise = big_noise*(1e-16)

    sn = np.nanmean(np.divide(spaxel_flux,spaxel_noise))

    #mask emission lines
    #I think you'd be better off masking by wavelength, rather than
    #flux here.
    mask = (spaxel_flux < 1e-15) & (spaxel_flux > -1e-15)
    masked_flux = spaxel_flux[mask]
    masked_lam = lam[mask]

    #plot continuum
    finite = np.isfinite(masked_flux)
    cont = np.polyfit(masked_lam[finite], masked_flux[finite], 3)
    fit = np.poly1d(cont)

    # Gaussian fit to Pa alpha emission line
    g1 = models.Gaussian1D(2e-16, 2.03536419463, .001, \
                            bounds={'amplitude':[0,1e-10]})
    g2 = models.Gaussian1D(1e-16, 2.03362675026, 0.005,bounds={'amplitude':[0,1e-10]})
    gg_init = g1 + g2
    fitter = fitting.LevMarLSQFitter()
    gg_fit = fitter(gg_init, lam, np.subtract(spaxel_flux, fit(lam)))

    if plot == True:
        plt.figure(figsize=(8,5))
        plt.plot(lam, spaxel_flux, color = 'k')
        plt.plot(lam, gg_fit(lam) + fit(lam), color = 'r')
        plt.plot(lam, gg_fit[0](lam) + fit(lam), color = 'purple', linestyle = '--')
        plt.plot(lam, gg_fit[1](lam) + fit(lam), color = 'g', linestyle = '--')
    ##Plot continuum
        plt.plot(lam, fit(lam), color = 'b', linestyle = '--', linewidth = 1)
        plt.show()

    if np.all(np.isnan(fitter.fit_info['fjac'])) == True:
        print('No covariance')
        gg_fit = 0
    else:
        print('OK')

    return gg_fit

#--------------------------------------------------------------------------------------------------------------------------

def main():

    z, lam, flux = data('COADD_mean3sig.fits')

    #Estimate for position of Pa alpha line
    #Pa alpha rest frame wavelength = 1.87 micons
    #Pa = 1.87 + (z*1.87)
    Pa_region = (lam > 2.034-0.015) & (lam < 2.034+0.015)

    #Pa_central_fit = spaxel_fit(lam[Pa_region], flux[Pa_region,34,32], 2e-15)
    #Pa_central_amp = (Pa_central_fit[0].amplitude)[0]

    #xx=spaxel_gauss_fit = spaxel_fit(lam[Pa_region], flux[Pa_region,6,60], True)

    stack = np.sum(np.sum(flux[Pa_region,2:8,58:63], axis=1),axis=1)
    plt.figure(figsize=(8,5))
    plt.plot(lam[Pa_region], stack, color = 'k')
    plt.show()
    print flux.shape
    print stack.shape
    quit()
    #array to store data
    parameters = np.zeros((67,64,6))
    bestfit = np.zeros((67,64,len(lam[Pa_region])))
    lineflux = np.zeros((67,64))

    #Run function to fit emission lines for all spaxels.
    bad = 0
    for x in range(0,67):
        for y in range(0,64):
            print x, y
            spaxel_gauss_fit = spaxel_fit(lam[Pa_region], flux[Pa_region,x,y])

            if np.all(spaxel_gauss_fit==0) == True:
                print('nan')
                parameters[x,y,:] = np.nan
                bad = bad+1
            else:
                parameters[x,y,0] = (spaxel_gauss_fit[0].amplitude)[0]
                parameters[x,y,1] = (spaxel_gauss_fit[0].mean)[0]
                parameters[x,y,2] = (spaxel_gauss_fit[0].stddev)[0]

                parameters[x,y,3] = (spaxel_gauss_fit[1].amplitude)[0]
                parameters[x,y,4] = (spaxel_gauss_fit[1].mean)[0]
                parameters[x,y,5] = (spaxel_gauss_fit[1].stddev)[0]

                bestfit[x,y,:] = spaxel_gauss_fit(lam[Pa_region])
                lineflux[x,y] = np.sqrt(2*np.pi)*\
                                (parameters[x,y,0]*parameters[x,y,1]+\
                                 parameters[x,y,3]*parameters[x,y,4])

    hdu2 = fits.PrimaryHDU()
    hdu2.data = bestfit
    hdu2.writeto('Pa_alpha_BestFit.fits', clobber=True)

    hdu2 = fits.PrimaryHDU()
    hdu2.data = parameters
    hdu2.writeto('Pa_alpha_Params.fits', clobber=True)

    hdu2 = fits.PrimaryHDU()
    hdu2.data = lineflux
    hdu2.writeto('Pa_alpha_Flux.fits', clobber=True)

main()
