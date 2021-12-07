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
import os
import numpy as np

#NOTE: when used SPECTRUM for the synthesis, the strong line (reported 
#in the SPECTRUM's files like strong6.c, balmer8.c, paschen3.c) must not 
#be included in the line list,otherwise they will be computed double time.
#
#NOTE: in the file stdatom.dat (used by SPECTRUM) are reported the number of
#ionizations allowed by SPECTRUM. One line that has another ionization level
#will no be computed. This behaviour is taken in account and hard coded
#into the present version of the calibrator code.

list_pars = [] # do not edit
##########################################
# edit directories
home_dir = os.getenv("HOME")
working_dir = home_dir + './'
calibrated_file = working_dir + 'llist_table'

atm_dir = working_dir + 'atmospheres/'
file_ll_VALD = working_dir + 'VALD/VALD_4800-6860_norep.dat'
file_ll_hyperfine = working_dir + 'VALD/hyp_fine_split/gfhy_0500-1200.dat'
stdatom_file = working_dir + 'stdatom.dat'
file_hfs_Co = working_dir + 'llist_sneden/llists/Co1.dat'
file_hfs_Mn = working_dir + 'llist_sneden/llists/Mn12.dat'
file_hfs_V = working_dir + 'llist_sneden/llists/V1.dat'
list_hfs = [file_hfs_Co, file_hfs_Mn, file_hfs_V]
list_luke = working_dir + 'llist_spectrum/luke.dat'
##########################################
#edit how many spectra to use
n_spec = 5
color_stars=['orange','tomato','royalblue','forestgreen','blueviolet']
##########################################
def sigma_disp(name_star,wave):

    Wgrid = np.array([4800., 5300., 5800., 6300., 8400.])
    #these are the sigma (resolution for star and wvalenegth interval)
    sigma_Sun = {4800: 0.035, 5300: 0.05, 5800: 0.06, 6300: 0.06, 8400: 0.08}
    sigma_Arc = {4800: 0.06, 5300: 0.07, 5800: 0.07, 6300: 0.07, 8400: 0.08} 
    sigma_Pro = {4800: 0.08, 5300: 0.09, 5800: 0.10, 6300: 0.10, 8400: 0.11}    
    sigma_eVir = {4800: 0.07, 5300: 0.075, 5800: 0.085, 6300: 0.095, 8400: 0.11}    
    sigma_eEri = {4800: 0.05, 5300: 0.05, 5800: 0.04, 6300: 0.07, 8400: 0.13}    



    mask = (wave-Wgrid)>=0
    Wpoint = Wgrid[mask][-1]

    if name_star=='Sun':
        sigma = sigma_Sun[Wpoint]
    elif name_star=='Arcturus':
        sigma = sigma_Arc[Wpoint]
    elif name_star=='eEri':
        sigma = sigma_eEri[Wpoint]
    elif name_star=='Procyon':
#        sigma = 0.05+(wave-5500.)*0.8e-5
        sigma = sigma_Pro[Wpoint]
    elif name_star=='eVir':
        sigma = sigma_eVir[Wpoint]
    return sigma
##########################################
# edit parameters star0

name_star = 'Sun'
file_obs_sp = working_dir + 'std_spectra/solar_deg.dat'
file_atm_model = atm_dir + 'ap00-t5777-g44.mod'
microt = '1.0'# '1.0'
macrot = '2.5'
#sigma_disp = '0.04'#'0.04' for interval 4800-6860, '0.08' for interval 8400-8920
pix_step = '0.01'
mh = 0.0 #metallicity of the atmosphere
abd_dict = {'Fe': 0.0} #abundances as difference from the Sun abundances


# do not edit the next 5 lines
name_obj = 'star0'
file_abd_atm = atm_dir + 'abd_' + name_obj + '.dat'
file_synt_out = working_dir + 'synt_' + name_obj
file_ews_out = working_dir + 'ews_' + name_obj
list_pars.append([name_obj, name_star, file_obs_sp, file_abd_atm, file_atm_model, file_synt_out, file_ews_out, microt, macrot, sigma_disp, pix_step, mh, abd_dict])
##########################################
# edit parameters star1

name_star = 'Arcturus'

file_obs_sp = working_dir + 'std_spectra/arcturus_deg.dat'
file_atm_model = atm_dir + 'am035-t4286-g166.mod'
microt = '1.6'#'1.7' '1.5'
macrot = '3.5'#'5.3'
#sigma_disp = '0.04'
pix_step = '0.01'
mh = -0.35
#abundances as difference from the Sun abundances
#original abd (from Jofre' et al, A&A 582, 81, 2015)
abd_dict = {'Mg':-0.16, 'Si':-0.25, 'Ca':-0.40, 'Sc':-0.43, 'Ti':-0.31, 'V':-0.44, 'Cr':-0.58, 'Mn':-0.89, 'Fe': -0.52, 'Co':-0.41,'Ni':-0.49} 

# do not edit the next 5 lines
name_obj = 'star1'
file_abd_atm = atm_dir + 'abd_' + name_obj + '.dat'
file_synt_out = working_dir + 'synt_' + name_obj
file_ews_out = working_dir + 'ews_' + name_obj
list_pars.append([name_obj, name_star, file_obs_sp, file_abd_atm, file_atm_model, file_synt_out, file_ews_out, microt, macrot, sigma_disp, pix_step, mh, abd_dict])
##########################################
# edit parameters star2

name_star = 'Procyon'

file_obs_sp = working_dir + 'std_spectra/procyon_deg.dat'
file_atm_model = atm_dir + 'am004-t6554-g399.mod'
microt = '1.7'#'2.1' '1.7'
macrot = '6.5'#'7.5'
#sigma_disp = '0.05'
pix_step = '0.01'
mh = -0.04
#abundances as difference from the Sun abundances
#original abd (from Jofre' et al, A&A 582, 81, 2015)
abd_dict = {'Mg':-0.04, 'Si':-0.03, 'Ca':0.04, 'Sc':-0.13, 'Ti':-0.07, 'V':-0.14, 'Cr':-0.12, 'Mn':-0.12, 'Fe': 0.01, 'Co':-0.10,'Ni':-0.11} 

# do not edit the next 5 lines
name_obj = 'star2'
file_abd_atm = atm_dir + 'abd_' + name_obj + '.dat'
file_synt_out = working_dir + 'synt_' + name_obj
file_ews_out = working_dir + 'ews_' + name_obj
list_pars.append([name_obj, name_star, file_obs_sp, file_abd_atm, file_atm_model, file_synt_out, file_ews_out, microt, macrot, sigma_disp, pix_step, mh, abd_dict])
##########################################
# edit parameters star3

name_star = 'eVir'

file_obs_sp = working_dir + 'std_spectra/NARVAL_epsVir.dat'
file_atm_model = atm_dir + 'ap015-t4983-g277.mod'
microt = '1.3'#'1.5'
macrot = '5.0'#'6.0'
#sigma_disp = '0.05' #'0.05' for interval 4800-6860, '0.07' for interval 8400-8920
pix_step = '0.01'
mh = 0.15
#abundances as difference from the Sun abundances
#original abd (from Jofre' et al, A&A 582, 81, 2015)
abd_dict = {'Mg':0.06, 'Si':0.18, 'Ca':0.10, 'Sc':0.06, 'Ti':-0.02, 'V':-0.03, 'Cr':0.06, 'Mn':-0.12, 'Fe':0.15, 'Co':-0.03,'Ni':0.09} 

# do not edit the next 5 lines
name_obj = 'star3'
file_abd_atm = atm_dir + 'abd_' + name_obj + '.dat'
file_synt_out = working_dir + 'synt_' + name_obj
file_ews_out = working_dir + 'ews_' + name_obj
list_pars.append([name_obj, name_star, file_obs_sp, file_abd_atm, file_atm_model, file_synt_out, file_ews_out, microt, macrot, sigma_disp, pix_step, mh, abd_dict])
##########################################
# edit parameters star4

name_star = 'eEri'
file_obs_sp = working_dir + 'std_spectra/UVES.POP.S4N_epsEri.dat'
file_atm_model = atm_dir + 'am009-t5050-g460.mod'
microt = '1.0'
macrot = '3.5'
#sigma_disp = '0.05' #'0.05' for interval 4800-6860, '0.13' for interval 8400-8920
pix_step = '0.01'
mh = -0.09
#abundances as difference from the Sun abundances
#original abd (from Jofre' et al, A&A 582, 81, 2015)
abd_dict = {'Mg':-0.08, 'Si':-0.09, 'Ca':-0.05, 'Sc':-0.16, 'Ti':-0.04, 'V':-0.02, 'Cr':-0.03, 'Mn':-0.16, 'Fe':-0.09, 'Co':-0.20,'Ni':-0.18} 

# do not edit the next 5 lines
name_obj = 'star4'
file_abd_atm = atm_dir + 'abd_' + name_obj + '.dat'
file_synt_out = working_dir + 'synt_' + name_obj
file_ews_out = working_dir + 'ews_' + name_obj
list_pars.append([name_obj, name_star, file_obs_sp, file_abd_atm, file_atm_model, file_synt_out, file_ews_out, microt, macrot, sigma_disp, pix_step, mh, abd_dict])
###########################################