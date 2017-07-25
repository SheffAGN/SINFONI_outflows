from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from scipy.stats import chi2

def data(file):

    z = 0.085

    hdulist = fits.open(file)

    hdulist.info()
    hdr = hdulist[0].header
    #print(hdr)
    flux = hdulist[0].data
    print(flux.shape)

    hdulist.close()

    return z, flux
    
def wavelength(file):
    
    hdulist = fits.open(file)
    hdulist.info()
    hdr = hdulist[0].header
    
    #lam = np.arange(float(hdr['CRVAL3']), float(hdr['CRVAL3'])+float(hdr['NAXIS3'])*float(hdr['CDELT3']), float(hdr['CDELT3']))
    lam = np.arange(float(hdr['CRVAL3'])-(float(hdr['NAXIS3'])/2)*float(hdr['CDELT3']), float(hdr['CRVAL3'])+(float(hdr['NAXIS3'])/2-0.5)*float(hdr['CDELT3']), float(hdr['CDELT3']))
    print(len(lam))
    
    hdulist.close()
    return lam

#--------------------------------------------------------------------------------------------------------------------------


def spaxel_fit(inlam, influx, plot=True):
    n = 0
    if np.all(influx==0) == True:
        print('zeros')
        #return 0.
        gg_fit = 0
        
    if np.all(np.isnan(influx)==True):
        print('zeros')
        #return 0.
        gg_fit = 0
       
    else:
        o = ~np.isnan(influx)
        lam = inlam[o]
        spaxel_flux = influx[o]
        
        ##remove emission lines to fins noise estimate
        #new_flux = spaxel_flux
        #new_flux[np.where(new_flux > 1e-15)] = np.nan
        #new_flux[np.where(new_flux < -1e-15)] = np.nan
        ##spaxel_flux[np.where(spaxel_flux == 0)] = np.nan
        #
        #big_flux = new_flux*(1e16)
        #big_noise = np.sqrt(big_flux)
        #spaxel_noise = big_noise*(1e-16)
        #
        #sn = np.nanmedian(np.divide(new_flux,spaxel_noise))
        
        
        #mask emission lines
        #I think you'd be better off masking by wavelength, rather than
        #flux here.
        #mask = (spaxel_flux < 1e-15) & (spaxel_flux > -1e-15)
        
        #mask = np.logical_or((lam < 2.032), (lam > 2.04))
        mask = np.logical_or((lam < 2.01), (lam > 2.05))
        masked_flux = spaxel_flux[mask]
        masked_lam = lam[mask]
      
        #plot continuum
        finite = np.isfinite(masked_flux)
        #cont, res, _, _, _ = np.polyfit(masked_lam[finite], masked_flux[finite], 3, full = True)
        cont = np.polyfit(masked_lam[finite], masked_flux[finite], 1) 
        #cont = np.polyfit(masked_lam[finite], masked_flux[finite], 3)
        fit = np.poly1d(cont)
        
        #print(len(masked_flux))
        #print(len(fit(masked_lam)))
        err = np.sqrt(np.sum((masked_flux - fit(masked_lam))**2)/len(masked_flux))
        #print(err)
        
        #Chi-squared before fit
        chi0 = np.sum((spaxel_flux - fit(lam))**2)/err
        
        
        # Gaussian fit to Pa alpha emission line
        #g1 = models.Gaussian1D(2e-16, 2.03536419463, .001, \
        #                        bounds={'amplitude':[0,1e-10]})
        #g2 = models.Gaussian1D(1e-16, 2.03362675026, 0.005,bounds={'amplitude':[0,1e-10]})
        
        g1 = models.Gaussian1D(6000, 2.03536419463, .001, \
                                bounds={'amplitude':[0,1e10]})
        g2 = models.Gaussian1D(2000, 2.03362675026, 0.005,bounds={'amplitude':[0,1e10]})
        
        gg_init = g1 + g2
        fitter = fitting.LevMarLSQFitter()
        gg_fit = fitter(gg_init, lam, np.subtract(spaxel_flux, fit(lam)))
        #stddev1 = (gg_fit[0].stddev)[0]
        
        #Chi-squared after fit for single gaussian
        chi1_sqd = np.sum((spaxel_flux - (gg_fit[0](lam)+fit(lam)))**2)/err
        delt_chi1 = chi0 - chi1_sqd
        #Chi-squared after fit for compund gaussian
        chi2_sqd = np.sum((spaxel_flux - (gg_fit(lam)+fit(lam)))**2)/err
        delt_chi2 = chi0 - chi2_sqd
        #print('HERE')
        #print(gg_fit[0].amplitude)
        #print(gg_fit[1].amplitude)
        
        if plot == True:
            plt.figure(figsize=(8,5))
            plt.plot(lam, spaxel_flux, color = 'k')
            plt.plot(lam, gg_fit(lam) + fit(lam), color = 'r')
            plt.plot(lam, gg_fit[0](lam) + fit(lam), color = 'purple', linestyle = '--')
            plt.plot(lam, gg_fit[1](lam) + fit(lam), color = 'g', linestyle = '--')
        ##Plot continuum
            plt.plot(lam, fit(lam), color = 'b', linestyle = '--', linewidth = 1)
            plt.show()
        
     
        #if fitter.fit_info['param_cov'] == None:
        if np.all(np.isnan(fitter.fit_info['fjac'])) == True:
            print('No covariance')
            gg_fit = 0
            
        
        conf1 = chi2.ppf(0.9973, 3)
        conf2 = chi2.ppf(0.9973, 6)
        #print(conf)
        
    
        if delt_chi1 < conf1:
            print('fitting noise')
            gg_fit = 0
            
            
        else:
            #print('ok')
            if delt_chi2 < conf2:
                gg_init = g1
                fitter = fitting.LevMarLSQFitter()
                gg_fit = fitter(gg_init, lam, np.subtract(spaxel_flux, fit(lam)))
                n = 1
                
      
            
        #if stddev1 < 1e-4:
        #    print('Small stddev')
        #    gg_fit = 0
        #
        #if sn < 2:
        #    print('bad sn ratio')
        #    gg_fit = 0
        #
        #else:
        #    print('OK')
    
    return gg_fit, n

#--------------------------------------------------------------------------------------------------------------------------

def main():

    z, flux = data('Stacked_observations.fits')
    #print(flux)
    print(flux[:,0,1])
    lam = wavelength('/Users/charlotteavery/Documents/SURE_project/Teacup/1327202/cor/out_cube_obj_cor00.fits')
    print(lam)
    #help(lam)
    #Estimate for position of Pa alpha line
    #Pa alpha rest frame wavelength = 1.87 micons
    #Pa = 1.87 + (z*1.87)
    #Pa_region = (lam > 2.034-0.015) & (lam < 2.034+0.015)
    Pa_region = (lam > 2.034-0.04) & (lam < 2.034+0.04)
    
    #plt.plot(lam[Pa_region], flux[:,35,35][Pa_region])
    #plt.show()
    
    #Pa_central_fit = spaxel_fit(lam[Pa_region], flux[Pa_region,34,32], 2e-15)
    #Pa_central_amp = (Pa_central_fit[0].amplitude)[0]

    #xx=spaxel_gauss_fit = spaxel_fit(lam[Pa_region], flux[Pa_region,6,60], True)

    #stack = np.sum(np.sum(flux[Pa_region,2:8,58:63], axis=1),axis=1)
    #plt.figure(figsize=(8,5))
    #plt.plot(lam[Pa_region], stack, color = 'k')
    #plt.show()
    #print (flux.shape)
    #print (stack.shape)
    #quit()
    
    spaxel_gauss_fit, n = spaxel_fit(lam[Pa_region], flux[Pa_region,20,45])
    quit()
    
    #array to store data
    parameters = np.zeros((68,72,6))
    bestfit = np.zeros((68,72,len(lam[Pa_region])))
    lineflux = np.zeros((68,72))

    #Run function to fit emission lines for all spaxels.
    bad = 0
    for x in range(0,68):
        for y in range(0,72):
            print (x, y)
            spaxel_gauss_fit, n = spaxel_fit(lam[Pa_region], flux[Pa_region,x,y])
            print(spaxel_gauss_fit)
            
            if n == 0:
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
            
            if n == 1:
                if np.all(spaxel_gauss_fit==0) == True:
                    print('nan')
                    parameters[x,y,:] = np.nan
                    bad = bad+1
                
                 
                else:
                    parameters[x,y,0] = (spaxel_gauss_fit.amplitude)[0]
                    parameters[x,y,1] = (spaxel_gauss_fit.mean)[0]
                    parameters[x,y,2] = (spaxel_gauss_fit.stddev)[0]
                
                    parameters[x,y,3] = np.nan
                    parameters[x,y,4] = np.nan
                    parameters[x,y,5] = np.nan

                    bestfit[x,y,:] = spaxel_gauss_fit(lam[Pa_region])
                    lineflux[x,y] = np.sqrt(2*np.pi)*\
                                    (parameters[x,y,0]*parameters[x,y,1])
                

    hdu2 = fits.PrimaryHDU()
    hdu2.data = bestfit
    #hdu2.writeto('Pa_alpha_BestFit_test.fits', clobber=True)

    hdu2 = fits.PrimaryHDU()
    hdu2.data = parameters
    #hdu2.writeto('Pa_alpha_Params_test.fits', clobber=True)

    hdu2 = fits.PrimaryHDU()
    hdu2.data = lineflux
    hdu2.writeto('Pa_alpha_Flux_deepimg3.fits', clobber=True)

main()