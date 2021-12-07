#    llist_calibrator performs an empirical calibration of the
#    transition probabilities of spectral lines by using observed
#    spectra of standard stars and a software for stellar spectrum synthesis.
#
#    Copyright (C) 2019 Corrado Boeche
#    
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#####################################################################

#####################################################################


import os, sys
import numpy as np
from scipy.optimize import leastsq,least_squares
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import pdb

import calibrator_utils as utils
import calibrator_classes as classes

############################################
def set_observed_spectra_intervals(obj_list):

    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        processes.append(mp.Process(target=obj.set_obs_sp_interval, args=[output, pos]))
    
    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()
    for i,r in enumerate(result):
        obj_list[i].observed_sp_df=r[1]


    # Exit the completed processes
    for p in processes:
        p.join()

    # terminate the processes
    for p in processes:
        p.terminate()

######################################
def synthesize_orig_spectra(obj_list):

    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        processes.append(mp.Process(target=obj.synthesize, args=[output,pos]))
    
    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()

    for i,r in enumerate(result):
        obj_list[i].ini_synthetic_sp_df=r[1]

    # Exit the completed processes
    for p in processes:
        p.join()


    # terminate the processes
    for p in processes:
        p.terminate()

######################################
def synthesize_spectra(obj_list, mode=None):

    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        if mode==None:
            processes.append(mp.Process(target=obj.synthesize, args=[output,pos]))
        elif mode=='strong':
            processes.append(mp.Process(target=obj.synthesize_strong, args=[output,pos]))

    
    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()

    for i,r in enumerate(result):
        if mode==None:
            obj_list[i].synthetic_sp_df=r[1]
        elif mode=='strong':
            obj_list[i].synth_strong_sp_df=r[1]

    # Exit the completed processes
    for p in processes:
        p.join()


    # terminate the processes
    for p in processes:
        p.terminate()


############################################
############################################
def normalize_spectra(obj_list):

    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        processes.append(mp.Process(target=obj.normalize, args=[output, pos]))

    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()

    for i,r in enumerate(result):
        obj_list[i].observed_sp_df=r[1]

    # Exit the completed processes
    for p in processes:
        p.join()


    # terminate the processes
    for p in processes:
        p.terminate()
############################################
def compute_equivalent_widths(obj_list, llist):

    #write the calib linelist
    llist.write_calibrating_ll()
    
    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        processes.append(mp.Process(target=obj.compute_ews, args=[output, pos]))
    
    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()
    for i,r in enumerate(result):
        #assign the index to r[1]
        r[1].index = llist.calib_indexes
        #remove two columns
        r[1].drop('wavelength', axis=1, level=1, inplace=True)
        r[1].drop('atom', axis=1, level=1, inplace=True)
        #save the values in llist.calibrated_full_ll_df
        llist.calibrated_full_ll_df.update(r[1])
    # Exit the completed processes
    for p in processes:
        p.join()


    # terminate the processes
    for p in processes:
        p.terminate()
############################################
def compute_newr(obj_list, llist):

    #write the calib linelist
#    llist.write_calibrating_ll()
    
    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        processes.append(mp.Process(target=obj.compute_newr, args=[output, pos]))
    
    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()
    for i,r in enumerate(result):
        #assign the index to r[1]
        df = r[1]
#        print(r[0], df)

        llist.calibrated_full_ll_df.update(df)
    # Exit the completed processes
    for p in processes:
        p.join()


    # terminate the processes
    for p in processes:
        p.terminate()

############################################
def residuals_spectra(obj_list):

    output = mp.SimpleQueue()
    processes=[]

    for pos, obj in enumerate(obj_list):
        processes.append(mp.Process(target=obj.compute_residuals, args=[output, pos]))
    
    # Run processes
    for p in processes:
        p.start()


    # Get process results from the output queue
    result = [output.get() for p in processes]
    result.sort()
    for i,r in enumerate(result):
        obj_list[i].residuals_sp_ds=r[1]

    # Exit the completed processes
    for p in processes:
        p.join()


    # terminate the processes
    for p in processes:
        p.terminate()

################################################
def compute_residuals(vars_loggf, obj_list, llist):

    loggf_df=pd.Series(vars_loggf)
    # limit the loggf between -10. and +3 
    # (if loggf is too small or too large,  the synthesis can fail)
#    loggf_df=loggf_df.apply(lambda x: max(-10.,min(3.,x)))
    #set the newly calibrated log gfs
    loggf_df.index=llist.calib_indexes
    llist.calibrated_full_ll_df.loc[llist.calib_indexes,('atomic_pars','loggf')] = loggf_df
    #write the linelist
    llist.write_synt_ll()

    synthesize_spectra(obj_list)
    normalize_spectra(obj_list)
    residuals_spectra(obj_list)

    #concatenate the residuals
    resid_list=[]
    for obj in obj_list:
        resid_list = resid_list + obj.residuals_sp_ds.tolist()
    residuals_all=np.asarray(resid_list)

    return residuals_all
################################################
def compute_VdW_residuals(VdW, obj_list, llist, indexes):

    #set the newly calibrated log gfs
    llist.calibrated_full_ll_df.loc[indexes,('atomic_pars','Waals')] = VdW
    #write the linelist
    llist.write_synt_ll()

    synthesize_spectra(obj_list)
    residuals_spectra(obj_list)

    #concatenate the residuals
    resid_list=[]
    for obj in obj_list:
        resid_list = resid_list + obj.residuals_sp_ds.tolist()
    residuals_all=np.asarray(resid_list)

    return residuals_all
#############################################
def minimize_spectra(obj_list, llist):


    #define the variables
    vars_loggf =llist.calibrated_full_ll_df.loc[llist.calib_indexes, ('atomic_pars','loggf')].values
    if len(vars_loggf)>0:
        #compute the least squares
        out = least_squares(compute_residuals, vars_loggf, args=(obj_list, llist), bounds=(-12.,3.), method='trf',diff_step=0.05, ftol=1e-6)

        #write the newly computed log gfs
        loggf_df=pd.Series(out.x)
        loggf_df=loggf_df.round(2)
        #set the same indexes of llist.calibrating_interval_df
        loggf_df.index=llist.calib_indexes
        #set the newly calibrated log gfs
        llist.calibrated_full_ll_df.loc[llist.calib_indexes,('atomic_pars','loggf')] = loggf_df

    #re-synthesize the spectra
    synthesize_spectra(obj_list)
    #compute the new residuals
    residuals_spectra(obj_list)

#############################################
def minimize_VdW_spectra(obj_list, llist, indexes):


    #define the variables
    VdW =llist.calibrated_full_ll_df.loc[indexes, ('atomic_pars','Waals')]
    #compute the least squares
    out = least_squares(compute_VdW_residuals, VdW, args=(obj_list, llist, indexes), bounds=(0.1,10.), method='trf',diff_step=0.1, ftol=1e-6)

    #write the newly computed log gfs
    VdW_df=pd.Series(out.x)
    VdW_df=VdW_df.round(2)
    
    #set the same indexes of llist.calibrating_interval_df
    idx=pd.Index([indexes])
    #set the newly calibrated log gfs
    llist.calibrated_full_ll_df.loc[idx,('atomic_pars','Waals')] = VdW_df.iloc[0]
##############################################
def compute_purity(obj_list, llist):

    #create a list of indexes that contains indexes of calibrating and strong lines (if any) together
    calib_strong_indexes = llist.calib_indexes.union(llist.strong_indexes)

    #create a boolean array with length equal to calib_strong_indexes
    boole = llist.calibrated_full_ll_df.loc[calib_strong_indexes,('atomic_pars','wavelength')] > 0.
#    pdb.set_trace()
    for idx in calib_strong_indexes:
        #boolean to exclude one line per loop
        boole = llist.synt_indexes != idx 
        #write the llist with one line out
        #if the line idx is a normal line then
        if idx in llist.calib_indexes:
            with open(llist.ll_name, 'w') as outfile:
                llist.calibrated_full_ll_df.loc[llist.synt_indexes[boole],('atomic_pars',llist.par_names)].to_string(outfile,formatters=llist.format, index=False, header=False)
        else: #otherwise, if the line is a strong line
            with open(llist.ll_name, 'w') as outfile:
                llist.calibrated_full_ll_df.loc[llist.synt_indexes[~boole],('atomic_pars',llist.par_names)].to_string(outfile,formatters=llist.format, index=False, header=False)
        #synthesize the spectra with one line out (or with the strong line only, if idx is a strong line)
        synthesize_spectra(obj_list)
        for obj in obj_list:
            wave_line = llist.calibrated_full_ll_df.loc[idx,('atomic_pars','wavelength')]
            #fix a mimumum ew=1mA so that the wmin and wmax are not too close
            ew = np.max([0.001,llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'EW')]/1000.])
            sigma_dispersion = obj.sigma_disp(obj.name_star,wave_line)
            sigma = np.sqrt(sigma_dispersion**2 + (float(obj.macrot)*wave_line/obj.light_speed)**2 + (float(obj.microt)*wave_line/obj.light_speed)**2)
            #compute the strength_contr
            spec_wave = obj.synthetic_sp_df.wavelength
            spec_flux = obj.synthetic_sp_df.flux
            strength_but_one = obj.compute_strength(spec_wave, spec_flux, wave_line)
            #in some cases "strength_contr" can be a very small negative. Set to zero the minimum value.
            #if the idx line is a normal line, then
            if idx in llist.calib_indexes:
                llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'strength_contr')] = llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'strength_synt')] - strength_but_one
            else: #otherwise, if the line is a strong line do the following
                  #(in this case, the strength_but_one is the strength of the strong line only)
                llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'strength_contr')] = strength_but_one

            #compute the purity over 
            wmin, wmax = classes.compute_line_integration_interval(wave_line, sigma, ew)
            integrated_flux_but_one = obj.integrate_absorbed_flux(wmin, wmax, obj.synthetic_sp_df.wavelength, obj.synthetic_sp_df.flux)

            #if the idx line is a normal line, then
            if idx in llist.calib_indexes:
                llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'purity')] = (llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'abs_synt_flux')]- integrated_flux_but_one) / llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'abs_synt_flux')]
            else: #otherwise, if the line is a strong line do the following
                  #(in this case, the integrated_flux_but_one is the flux of the strong line only)
                llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'purity')] = integrated_flux_but_one / llist.calibrated_full_ll_df.loc[idx,(obj.name_obj,'abs_synt_flux')]
    #because now the obj.synthetic_sp_df and the file 'linelist' miss one line
    #let's re-synthesize them with the whole line list
    llist.write_synt_ll()
    synthesize_spectra(obj_list)
    