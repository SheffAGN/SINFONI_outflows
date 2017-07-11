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

   
def central_spaxel_fit(Pa_region, z, lam, flux):
    spaxel_flux = flux[:,34,32]

    #remove outliers
    spaxel_flux[np.where(spaxel_flux >4e-15)] = np.nan
    spaxel_flux[np.where(spaxel_flux < -1e-15)] = np.nan
    spaxel_flux[np.where(spaxel_flux == 0)] = np.nan
    
    big_flux = spaxel_flux*(10**16)
    big_noise = np.sqrt(big_flux)
    spaxel_noise = big_noise*(10**-16)
    
    print('Sn')
    sn = np.nanmean(np.divide(spaxel_flux,spaxel_noise))
    print(sn)
    
    #mask emission lines
    mask = (spaxel_flux < 1e-15) & (spaxel_flux > -1e-15)
    masked_flux = spaxel_flux[mask]
    masked_lam = lam[mask]
    
    #plot continuum
    finite = np.isfinite(masked_flux)
    cont = np.polyfit(masked_lam[finite], masked_flux[finite], 3)
    fit = np.poly1d(cont)
    #plt.plot(lam, spaxel_flux, color='k', linewidth = .5)
    #plt.plot(lam, fit(lam), color = 'r', linestyle = '--', linewidth = 1)
    #plt.xlabel('Wavelength ($\mu$m)')
    #plt.ylabel('Flux')
    #plt.show()
    #plt.close()
    
    # Gaussian fit to Pa alpha emission line
    g1 = models.Gaussian1D(amplitude=2e-15, mean=2.03536419463, stddev=.001, bounds = {'mean': (2.033, 2.037)}) 
    g2 = models.Gaussian1D(1e-15, 2.03362675026, 0.005, bounds = {'mean': (2.033, 2.037)})
    gg_init = g1 + g2
    fitter = fitting.LevMarLSQFitter()
    gg_fit = fitter(gg_init, lam[Pa_region], np.subtract(spaxel_flux[Pa_region], fit(lam)[Pa_region]))
    
    print('HERE')
    print(fitter.fit_info)
    #print(fitter.fit_info['param_cov'][0,0])
    #print(fitter.fit_info['param_cov'])
    
    amp1 = (gg_fit[0].amplitude)[0]
    amp2 = (gg_fit[1].amplitude)[0]
    
   
    
    if np.all(np.isnan(fitter.fit_info['fjac'])) == True:
        print('No covariance')
        gg_fit = 0
        
    #if fitter.fit_info['param_cov'] == None:
    #    print('No covariance')
    #    gg_fit = 0
    
    if sn < .5:
        print('low sn')
        gg_fit = 0
    
        
    else:
        print('OK')
        
        plt.figure(figsize=(8,5))
        plt.plot(lam[Pa_region], spaxel_flux[Pa_region], color = 'k')
        #plt.plot(lam[Pa_region], spaxel_noise[Pa_region], color = 'b')
        
        plt.plot(lam[Pa_region], gg_fit(lam[Pa_region]) + fit(lam)[Pa_region], color = 'r')
        #plt.plot(lam[Pa_region], gg_fit(lam[Pa_region]), color = 'c')
        plt.plot(lam[Pa_region], gg_fit[0](lam[Pa_region]) + fit(lam)[Pa_region], color = 'purple', linestyle = '--')
        plt.plot(lam[Pa_region], gg_fit[1](lam[Pa_region]) + fit(lam)[Pa_region], color = 'g', linestyle = '--')
        #Plot continuum
        plt.plot(lam[Pa_region], fit(lam[Pa_region]), color = 'b', linestyle = '--', linewidth = 1)
        plt.show()
        plt.close
            
    #print(gg_fit(lam[Pa_region]))
     
    return gg_fit
    
        
        
        
#--------------------------------------------------------------------------------------------------------------------------


def spaxel_fit(Pa_region, Pa_central_amp, x, y, z, lam, flux):
    spaxel_flux = flux[:,x,y]
  
    if np.all(spaxel_flux==0) == True:
        gg_fit = 0
        print('zeros')
        #no = 0  
    else:
    
        #remove outliers
        spaxel_flux[np.where(spaxel_flux >4e-15)] = np.nan
        spaxel_flux[np.where(spaxel_flux < -1e-15)] = np.nan
        spaxel_flux[np.where(spaxel_flux == 0)] = np.nan
        
        
        big_flux = spaxel_flux*(10**16)
        big_noise = np.sqrt(big_flux)
        spaxel_noise = big_noise*(10**-16)
   
        sn = np.nanmean(np.divide(spaxel_flux,spaxel_noise))
        
        
        #mask emission lines
        mask = (spaxel_flux < 1e-15) & (spaxel_flux > -1e-15)
        masked_flux = spaxel_flux[mask]
        masked_lam = lam[mask]
        
        #plot continuum
        finite = np.isfinite(masked_flux)
        cont = np.polyfit(masked_lam[finite], masked_flux[finite], 3)
        fit = np.poly1d(cont)
        #plt.plot(lam, spaxel_flux, color='k', linewidth = .5)
        #plt.plot(lam, fit(lam), color = 'r', linestyle = '--', linewidth = 1)
        #plt.xlabel('Wavelength ($\mu$m)')
        #plt.ylabel('Flux')
        #plt.show()
        #plt.close()
        
        # Gaussian fit to Pa alpha emission line
        g1 = models.Gaussian1D(amplitude=2e-15, mean=2.03536419463, stddev=.001) 
        g2 = models.Gaussian1D(1e-15, 2.03362675026, 0.005)
        gg_init = g1 + g2
        fitter = fitting.LevMarLSQFitter()
        gg_fit = fitter(gg_init, lam[Pa_region], np.subtract(spaxel_flux[Pa_region], fit(lam)[Pa_region]))
        
        amp1 = (gg_fit[0].amplitude)[0]
        amp2 = (gg_fit[1].amplitude)[0]
        mean1 = (gg_fit[0].mean)[0]
        mean2 = (gg_fit[1].mean)[0]
        
        true = np.sum(np.greater(gg_fit(lam[Pa_region]), fit(lam)[Pa_region]))
        print(true)
        #print(mean1)
        #print(mean2)
        #print(np.absolute(mean2 - mean1))
        
        #print(fitter.fit_info)
        #print(fitter.fit_info['param_cov'][0,0])
        #print(fitter.fit_info['param_cov'])
        
        #if fitter.fit_info['param_cov'] == None:
        #    print('No covariance')
        #    gg_fit = 0
          
        #no = 2
    
        if np.all(np.isnan(fitter.fit_info['fjac'])) == True:
            print('No covariance')
            gg_fit = 0
            
    
            #gg_init = models.Gaussian1D(amplitude=2e-15, mean=2.03536419463, stddev=.001) 
            #fitter = fitting.LevMarLSQFitter()
            #gg_fit = fitter(gg_init, lam[Pa_region], np.subtract(spaxel_flux[Pa_region], fit(lam)[Pa_region]))
            #
            #if np.all(np.isnan(fitter.fit_info['fjac'])) == True:
            #    gg_fit = 0
            #    no = 0
            #    
            #else:
            #    no = 1
              
        
        #elif amp1 < 0 or amp2 < 0:
        #    print('Negative')
        #    gg_init = models.Gaussian1D(amplitude=2e-15, mean=2.03536419463, stddev=.001) 
        #    fitter = fitting.LevMarLSQFitter()
        #    gg_fit = fitter(gg_init, lam[Pa_region], np.subtract(spaxel_flux[Pa_region], fit(lam)[Pa_region]))
        #    
        #    amp_new = gg_fit.amplitude[0]
        # 
        #    if amp_new < 0:
        #        gg_fit =0
        #        no = 0
        #    else:
        #        no = 1
        #    
        #        
        #elif amp1 < 0.1*Pa_central_amp:
        #    print('Small amp')
        #    gg_fit = 0
        #    no = 0
        #
        #if sn < .5:
        #    print('low sn')
        #    gg_fit = 0
        #       
        #
        #elif np.absolute(mean2 - mean1) > 0.005:
        #    print('g1, g2 difference')
        #    
        #    gg_init = models.Gaussian1D(amplitude=2e-15, mean=2.03536419463, stddev=.001) 
        #    fitter = fitting.LevMarLSQFitter()
        #    gg_fit = fitter(gg_init, lam[Pa_region], np.subtract(spaxel_flux[Pa_region], fit(lam)[Pa_region]))
        #    
        #    no = 1
        # 
       
        else:
            print('OK')
            
    
            #plt.figure(figsize=(8,5))
            #plt.plot(lam[Pa_region], spaxel_flux[Pa_region], color = 'k')
            #plt.plot(lam[Pa_region], gg_fit(lam[Pa_region]) + fit(lam)[Pa_region], color = 'r')
            #plt.plot(lam[Pa_region], gg_fit[0](lam[Pa_region]) + fit(lam)[Pa_region], color = 'purple', linestyle = '--')
            #plt.plot(lam[Pa_region], gg_fit[1](lam[Pa_region]) + fit(lam)[Pa_region], color = 'g', linestyle = '--')
            ##Plot continuum
            #plt.plot(lam[Pa_region], fit(lam[Pa_region]), color = 'b', linestyle = '--', linewidth = 1)
            #plt.show()
            #plt.close
          

    return gg_fit
    


      

#--------------------------------------------------------------------------------------------------------------------------

def main():
    
    z, lam, flux = data('/Users/charlotteavery/Documents/SURE project/COADD_mean3sig.fits')
    
    
    #Estimate for position of Pa alpha line
    #Pa alpha rest frame wavelength = 1.87 micons
    #Pa = 1.87 + (z*1.87)
    Pa_region = (lam > 2.034-0.015) & (lam < 2.034+0.015)
    
    
   
    Pa_central_fit = central_spaxel_fit(Pa_region, z, lam, flux)

  
    #print(Pa_central_fit.__dict__)
    #
    #
    Pa_central_amp = (Pa_central_fit[0].amplitude)[0]
    
 
    print(Pa_central_fit.__dict__)
    #print(Pa_central_fit[0])    #key paramters for first guassian g1
    #print(Pa_central_fit[1])    #key paramters for second guassian g2
    #

    
    #array to store data
    parameters = np.zeros((67,64,6))
    
    bestfit = np.zeros((67,64,len(lam[Pa_region])))
    
    #Run function to fit emission lines for all spaxels.
    
    for x in range(0,67):
        for y in range(0,64):
            
            spaxel_gauss_fit = spaxel_fit(Pa_region, Pa_central_amp, x, y, z, lam, flux)
            
    
            #print('HERE')
            #print(x)
            #print(y)
            #print(Pa_central_fit.__dict__)
            #
            #
            if np.all(spaxel_gauss_fit==0) == True:
                print('nan')
                parameters[x,y,:] = np.nan
            
            
            #if no == 0:
            #    print('nan')
            #    parameters[x,y,:] = np.nan
            #    
            #elif no == 1:
            #    parameters[x,y,0] = (spaxel_gauss_fit.amplitude)[0]
            #    parameters[x,y,1] = (spaxel_gauss_fit.mean)[0]
            #    parameters[x,y,2] = (spaxel_gauss_fit.stddev)[0]
            #    
            #                        
            #    parameters[x,y,3] = np.nan
            #    parameters[x,y,4] = np.nan
            #    parameters[x,y,5] = np.nan
            #    
            #    bestfit[x,y,:] = spaxel_gauss_fit(lam[Pa_region])
                       
            #elif no == 2:
            else:
                parameters[x,y,0] = (spaxel_gauss_fit[0].amplitude)[0]
                parameters[x,y,1] = (spaxel_gauss_fit[0].mean)[0]
                parameters[x,y,2] = (spaxel_gauss_fit[0].stddev)[0]
                
                                    
                parameters[x,y,3] = (spaxel_gauss_fit[1].amplitude)[0]
                parameters[x,y,4] = (spaxel_gauss_fit[1].mean)[0]
                parameters[x,y,5] = (spaxel_gauss_fit[1].stddev)[0]
                
                bestfit[x,y,:] = spaxel_gauss_fit(lam[Pa_region])
            
         
          
    
    ##save array as fits file
    ##hdu = fits.PrimaryHDU()
    ##hdu.data = parameters
    ##hdu.writeto('/Users/charlotteavery/Documents/SURE project/Pa_alpha_gaussfit2.fits')
    #
    hdu2 = fits.PrimaryHDU()
    hdu2.data = bestfit
    hdu2.writeto('/Users/charlotteavery/Documents/SURE project/Pa_alpha_gaussfit_noconstraints.fits')
    
    
    

main()

