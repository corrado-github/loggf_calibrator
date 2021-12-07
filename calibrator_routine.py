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


import sys,os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from io import StringIO
from scipy.stats import pearsonr
from multiprocessing import Pool

import pdb

import calibrator_spectra as minim
import calibrator_utils as utils
import calibrator_classes as classes
import calibrator_params as params
import calibrator_strong as strong

from sklearn.neighbors import NearestNeighbors
import pickle

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
############################
class abd_spinbox:
    #define a class spinbox for the many elements to be set in tab2
    def __init__(self,labelframe, colonna, riga, var_text, var_abd):
        self.ele_text=StringVar(value=var_text)
        self.ele_value=DoubleVar(value=var_abd)
        inf=self.ele_value.get()-0.5
        sup=self.ele_value.get()+0.5
        self.entry_field = Spinbox(labelframe, from_=inf, to=sup, increment=0.01,width=6,textvariable=self.ele_value,state=NORMAL)
        self.entry_field.grid(column=colonna+1, row=riga, sticky=(W), padx=20, pady=2)

############################
def remove_calib_lines(*args):
    #function associated to the button 'remove lines from line list' in tab1

    ix=()
    ix = llist_calib_box.curselection()

    #remove the lines to the LineList class dataframes
    llist.remove_lines_update(ix)
    #update the llist_calib_box
    inserted_calib_llist.set(llist.calibrating_interval_list)

##########################################
def add_to_linelist(box, llist_obj):
    #function associated to the button 'add' in listboxes lf21, lf22, lf23 in tab1

    idx=()
    idx = list(box.curselection())

    #add the lines to the LineList class dataframes
    llist_obj.add_lines_update(idx)
    #update the llist_calib_box
    inserted_calib_llist.set(llist.calibrating_interval_list)
##########################################
def move_in_line():
    #function associated to the button 'from line list'

    ix=()
    ix = llist_calib_box.curselection()

    #remove the lines to the LineList class dataframes
    llist.move_in_line(llist.calib_indexes[ix])
    #update the Entry box
    edit_line.set(llist.editing_line)
##########################################
def move_out_line(*args):
    #function associated to the button 'to line list'

    #remove the lines to the LineList class dataframes
    llist.move_out_line(edit_line.get())
    #update the llist_calib_box
    inserted_calib_llist.set(llist.calibrating_interval_list)
    #empty the edit_line box
    edit_line.set([])
##########################################
def undo_VdW(*args):
    #function associated to the button 'undo VdW' in tab2

    ix=()
    ix = VdW_calib_box.curselection()

    indexes = llist.VdW_indexes[VdW_inf.get():VdW_sup.get()]
    #set VdW to 1.0 for the selected line
    llist.calibrated_full_ll_df.loc[indexes[ix],('atomic_pars','Waals')] = 1.0
    #update the llist_calib_box
    show_VdW_lines()
##########################################
def add_manual_line(*args):
    #function associated to the button 'add' in lf22 to add manually a line

    #remove the lines to the LineList class dataframes
    llist.add_manually_line(line_added_manually.get())
    #update the llist_calib_box
    inserted_calib_llist.set(llist.calibrating_interval_list)
    #empty the edit_line box
    line_added_manually.set([])
##########################################
def add_guess_line(*args):
    #function associated to the button 'add' in lf4 to add a guessed line

    idx=()
    idx = list(lines_guess_box.curselection())
    if len(idx)<1:
        return None
    line_to_add = lines_guess.get().replace('(\' ','\'').replace('\'','').replace(')','').split(',')[idx[0]]
    #remove the lines to the LineList class dataframes
    llist.add_manually_line(line_to_add)
    #update the llist_calib_box
    inserted_calib_llist.set(llist.calibrating_interval_list)
    #empty the edit_line box
#    line_added_manually.set([])
##########################################
def show_elements(*args):
    #function associated to the button 'show elements' in lf1_tab2

    llist.show_elements()
    ele_to_show = llist.calibrated_elements

    ele_list_name = []
    for ele in ele_to_show:
        idx_flag = (llist.calibrated_full_ll_df.loc[:,('atomic_pars','atom')].astype(int) == ele) &\
                   (llist.calibrated_full_ll_df.loc[:,('atomic_pars','comment')] == 'calib')
        idx_flag_2 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'EW')].apply(lambda x: True if (x>ew_minim.get()).any() else False, axis=1)
        idx_flag[idx_flag==True] = idx_flag_2
        string = ' (Z = ' + str(ele) + ', ' + str(idx_flag.sum()) + ' lines)'
        ele_list_name.append('%-04s' % utils.number_to_name_dict[ele] + '%-10s' % string)
    ele_names.set(ele_list_name)
##########################################
def guess_routine():
    #function associated to the button 'guess!' in lf4 to guess a line

    list_obs_flux = []
    wave = wave_guess.get()

    #parallelize the computation
    with Pool(processes=4) as pool:
        list_resid_ew = [pool.apply(obj.integrate_residuals, args=(wave,)) for obj in obj_star_list]
    list_resid_ew[1] = list_resid_ew[1]*0.5 #empiric correction because the Arcturus profile badly match a gaussian


    #upload the model
    ML_guess_model = pickle.load(open('ML_guess.model', 'rb'))
    ML_guess_idx = pd.read_csv('ML_guess.idx', header=None)

    predicted = ML_guess_model.predict([np.asarray(list_resid_ew)])
    idx_array = ML_guess_model.kneighbors([np.asarray(list_resid_ew)], n_neighbors=5, return_distance=False)
    idx_label = ML_guess_idx.iloc[idx_array[0]].values.flatten()
    guessed_df = llist.calibrated_full_ll_df.loc[idx_label]
    #set the right wavelength and hEv
    guessed_df[('atomic_pars','wavelength')] = wave
    guessed_df[('atomic_pars','hEv')] = guessed_df[('atomic_pars','lEv')] + 1.0e8/wave #compute the hEv from the wavelength of the line
    guessed_df[('atomic_pars','source')] = 'guessed'
    lines_guessed_str = classes.df_to_list_of_strings(guessed_df.loc[:,('atomic_pars',llist.par_names)],llist.format)
    lines_guess.set(lines_guessed_str)
##########################################
def write_abd_atm_files(*args):
    #this function writes the abundance files "star?.abd" with the abundances
    #indicated in the spinboxes of tab2
    for star in obj_star_list[1:]:
        for key, val in star.abd_dict.items():
            name_spinbox = star.name_star + key  + '_spinbox'
            star.abd_dict[key] = spinbox_abd_dict[name_spinbox].ele_value.get()

        #write the abd file read by SPECTRUM
        utils.write_SPECTRUM_abd_file(star.file_abd_atm, star.abd_dict)
        #write the abd file that can be edited by the used 
        utils.write_abd_file(star.file_abd, star.abd_dict)

##########################################
def recompute_newr(*args):
    #this function recompute the NEWR (only) of every line by using the abundances reported in
    #the spinboxes of tab2. Since it re-synthesize the whole spectra, it is very slow and
    #not so practical
     
    for i,row in df_w_intervals.iloc[counter:].iterrows():
    
        print('wave_start = ', row.wave_start, 'wave_step = ', row.wave_step)
        wave_start = row.wave_start
        wave_step = row.wave_step
        #prepare the synthetic and observed spectra in the gived wavelength interval
        prepare_synth_obs_spectra(wave_start,wave_step)
        #compute the EWs
        minim.compute_equivalent_widths(obj_star_list, llist)
        #compute the NEWRs
        minim.compute_newr(obj_star_list, llist)
        #compute the purity parameter
        minim.compute_purity(obj_star_list, llist)
##########################################
def plot_newrs(*args):
    #this function plot the NEWRs for the elements in tab2.
    #It can take no more than 4 elements per time. If more are
    #given, it neglects the elements after the 4th one.

    #clean the axes_abd
    for ax in axes_abd.reshape(-1):
        ax.clear()

    ele_list_number = []
    ele_list_name = []

    ix=()
    ix = ele_chose_box.curselection()


    for i in ix[0:4]:
        atomic_number = llist.calibrated_elements[i]
        ele_list_number.append(atomic_number)
        ele_list_name.append(utils.number_to_name_dict[atomic_number])

    
    color_stars = params.color_stars

    x_sup=3000
    x_inf=0.9
    y_sup=0.39
    y_inf=-0.39

    for i in np.arange(len(obj_star_list)):
        color_number = i % params.n_spec
        for j in np.arange(len(ele_list_number)):
            if i==0 :      
                axes_abd[i,j].set_title(utils.number_to_name_dict[ele_list_number[j]])
            axes_abd[i,j].annotate(obj_star_list[i].name_star, xy=(0.7, 0.8), xycoords='axes fraction', color=color_stars[color_number], fontsize=8)
            axes_abd[i,j].plot([x_inf,x_sup],[0.,0.], color='grey',linestyle=':')


            idx_flag =(llist.calibrated_full_ll_df.loc[:,('atomic_pars','atom')].astype(int)==ele_list_number[j]) &\
                  (llist.calibrated_full_ll_df.loc[:,('atomic_pars','comment')]=='calib')
            idx_flag_3 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'purity')].apply(lambda x: True if (x>purity_par.get()).any() else False, axis=1)
            idx_flag[idx_flag==True] = idx_flag_3

            bool_flag=idx_flag
            # plot the points
            if bool_flag.any():
                axes_abd[i,j].grid(color='lightgrey', which='major', linestyle='--', alpha=0.5)
                newr=llist.calibrated_full_ll_df.loc[bool_flag, (obj_star_list[i].name_obj,'NEWR')].values.astype(float)
                ews=llist.calibrated_full_ll_df.loc[bool_flag, (obj_star_list[i].name_obj,'EW')].values
                purity=llist.calibrated_full_ll_df.loc[bool_flag, (obj_star_list[i].name_obj,'purity')].values
                flag = (newr < y_sup) & (newr > y_inf)
                flag_upper = (newr >= y_sup)
                flag_lower = (newr <= y_inf)
                #plot the lines as points if inside the y limits
                ews_clipped = np.clip(ews, 1, 5000)
                axes_abd[i,j].scatter(ews_clipped[flag], newr[flag], color=color_stars[color_number], marker='.', alpha=0.3)
                #otherwise, plot them as arrow
                newr_clipped = np.clip(newr, y_inf+0.05, y_sup-0.05)
                axes_abd[i,j].scatter(ews_clipped[flag_upper], newr_clipped[flag_upper], color=color_stars[color_number], marker='^', alpha=1)
                axes_abd[i,j].scatter(ews_clipped[flag_lower], newr_clipped[flag_lower], color=color_stars[color_number], marker='v', alpha=1)
                #
                newr_std = np.std(newr)
                newr_avg = np.average(newr)
                bool_clip = (abs(newr)< 0.5) & (ews>5) 
                if bool_clip.sum()>3:
                    mean_str = 'mean=%+5.3f' % np.average(newr[bool_clip])
                    axes_abd[i,j].annotate(mean_str, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=8)
                    #
                    r_xy = 'r=%+5.3f' % pearsonr(ews[bool_clip], newr[bool_clip])[0] # [0] is the pearson correlation. [1] is the p-value
                    axes_abd[i,j].annotate(r_xy, xy=(0.5, 0.2), xycoords='axes fraction', fontsize=8)
                    #
                    N_str ='N={: 4d}'.format(bool_clip.sum())#   'N=%4d' % bool_flag.sum()
                    axes_abd[i,j].annotate(N_str, xy=(0.1, 0.1), xycoords='axes fraction', fontsize=8)
              
            if j==0:
                axes_abd[i,j].set_ylabel('NEWR')
            if i==len(obj_star_list)-1:
                axes_abd[i,j].set_xlabel(r'EW ($m\AA$)')

#            axes_abd[i,j].set_xticks(np.linspace(-11,0,12))

#            print(i,j,utils.number_to_name_dict[ele_list_number[j]],obj_star_list[i].name_obj,y_inf,y_sup)
            axes_abd[i,j].set_ylim(top=y_sup, bottom=y_inf)
            axes_abd[i,j].set_xlim(x_inf,x_sup)
            axes_abd[i,j].set_xscale('log')
#            axes_abd[i,j].set_yscale('symlog')

#            axes_abd[i,j].set_xticks([1,10,20,50,100,200,500,1000,2000,5000])

#    ####
#    plt.savefig('prova.pdf')
    canvas_abd.draw()

##########################################
def plot_params_check(*args):
    #this function is associated the the button "plot params check" in tab2
    #and it plots MOOG-like plots of NEWRs as a function of excitation potential,
    #reduced EW, and wavelength.

    #clean the axes_abd
    for ax in axes_abd.reshape(-1):
        ax.clear()

    ix=()
    ix = ele_chose_box.curselection()

    atomic_number = llist.calibrated_elements[ix[0]]
    ele_list_number = [atomic_number]
    ele_list_name = [utils.number_to_name_dict[atomic_number]]

    y_sup=0.39
    y_inf=-0.39

    
    color_stars = params.color_stars

    for i in np.arange(len(obj_star_list)):
        color_number = i % params.n_spec
        for j in np.arange(3):
            if i==0 :      
                axes_abd[i,j].set_title(ele_list_name)
            axes_abd[i,j].annotate(obj_star_list[i].name_star, xy=(0.1, 0.8), xycoords='axes fraction', color=color_stars[color_number], fontsize=8)
#            axes_abd[i,j].plot([x_inf,x_sup],[0.,0.], color='grey',linestyle=':')


            bool_flag =(llist.calibrated_full_ll_df.loc[:,('atomic_pars','atom')].astype(int)== atomic_number) &\
                  (llist.calibrated_full_ll_df.loc[:,('atomic_pars','comment')]=='calib') &\
                  (llist.calibrated_full_ll_df.loc[:,(obj_star_list[i].name_obj,'purity')]>purity_par.get())
            # plot the points
            if bool_flag.any():
                axes_abd[i,j].grid(color='lightgrey', which='major', linestyle='--', alpha=0.5)
                newr=llist.calibrated_full_ll_df.loc[bool_flag, (obj_star_list[i].name_obj,'NEWR')].values
                ews=llist.calibrated_full_ll_df.loc[bool_flag, (obj_star_list[i].name_obj,'EW')].values
                ews_red=np.log(ews/llist.calibrated_full_ll_df.loc[bool_flag, ('atomic_pars','wavelength')].values)
                purity=llist.calibrated_full_ll_df.loc[bool_flag, (obj_star_list[i].name_obj,'purity')].values
                EP = (llist.calibrated_full_ll_df.loc[bool_flag, ('atomic_pars','lEv')]*1.2398e-4).values
                wave = llist.calibrated_full_ll_df.loc[bool_flag, ('atomic_pars','wavelength')].values
                flag = (newr < y_sup) & (newr > y_inf)
                flag_upper = (newr >= y_sup)
                flag_lower = (newr <= y_inf)
                if j==1:
                    #plot the lines as points if inside the y limits
                    axes_abd[i,j].scatter(ews_red[flag], newr[flag], color=color_stars[color_number], marker='.', alpha=0.3)
                    #otherwise, plot them as arrow
                    newr_clipped = np.clip(newr, y_inf+0.05, y_sup-0.05)
                    axes_abd[i,j].scatter(ews_red[flag_upper], newr_clipped[flag_upper], color=color_stars[color_number], marker='^', alpha=1)
                    axes_abd[i,j].scatter(ews_red[flag_lower], newr_clipped[flag_lower], color=color_stars[color_number], marker='v', alpha=1)
                    axes_abd[i,j].set_xlabel(r'log(EW/$\lambda$)')
                    if (ews_red>-20).sum()>3: #plot the statistic only if there are more than three lines
                        mean_str = 'mean=%+5.3f' % np.average(newr)
                        axes_abd[i,j].annotate(mean_str, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=8)
                        #
                        r_xy = 'r=%+5.3f' % pearsonr(ews_red, newr)[0] # [1] is the p-value
                        axes_abd[i,j].annotate(r_xy, xy=(0.5, 0.2), xycoords='axes fraction', fontsize=8)
                        #
                        N_str ='N={: 4d}'.format(bool_flag.sum())#   'N=%4d' % bool_flag.sum()
                        axes_abd[i,j].annotate(N_str, xy=(0.1, 0.1), xycoords='axes fraction', fontsize=8)
                elif j==0:
                    #plot the lines as points if inside the y limits
                    axes_abd[i,j].scatter(EP[flag], newr[flag], color=color_stars[color_number], marker='.', alpha=0.3)
                    #otherwise, plot them as arrow
                    newr_clipped = np.clip(newr, y_inf+0.05, y_sup-0.05)
                    axes_abd[i,j].scatter(EP[flag_upper], newr_clipped[flag_upper], color=color_stars[color_number], marker='^', alpha=1)
                    axes_abd[i,j].scatter(EP[flag_lower], newr_clipped[flag_lower], color=color_stars[color_number], marker='v', alpha=1)                
                    axes_abd[i,j].set_xlabel('E.P.')
                    if (EP>0).sum()>3: #plot the statistic only if there are more than three lines[5~
                        mean_str = 'mean=%+5.3f' % np.average(newr)
                        axes_abd[i,j].annotate(mean_str, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=8)
                        #
                        r_xy = 'r=%+5.3f' % pearsonr(EP, newr)[0] # [1] is the p-value
                        axes_abd[i,j].annotate(r_xy, xy=(0.5, 0.2), xycoords='axes fraction', fontsize=8)
                        #
                        N_str ='N={: 4d}'.format(bool_flag.sum())#   'N=%4d' % bool_flag.sum()
                        axes_abd[i,j].annotate(N_str, xy=(0.1, 0.1), xycoords='axes fraction', fontsize=8)
                elif j==2:
                    #plot the lines as points if inside the y limits
                    axes_abd[i,j].scatter(wave[flag], newr[flag], color=color_stars[color_number], marker='.', alpha=0.3)
                    #otherwise, plot them as arrow
                    newr_clipped = np.clip(newr, y_inf+0.05, y_sup-0.05)
                    axes_abd[i,j].scatter(wave[flag_upper], newr_clipped[flag_upper], color=color_stars[color_number], marker='^', alpha=1)
                    axes_abd[i,j].scatter(wave[flag_lower], newr_clipped[flag_lower], color=color_stars[color_number], marker='v', alpha=1)
                    axes_abd[i,j].set_xlabel(r'lambda ($\AA$)')
                    if (EP>0).sum()>3: #plot the statistic only if there are more than three lines
                        mean_str = 'mean=%+5.3f' % np.average(newr)
                        axes_abd[i,j].annotate(mean_str, xy=(0.1, 0.2), xycoords='axes fraction', fontsize=8)
                        #
                        r_xy = 'r=%+5.3f' % pearsonr(EP, newr)[0] # [1] is the p-value
                        axes_abd[i,j].annotate(r_xy, xy=(0.5, 0.2), xycoords='axes fraction', fontsize=8)
                        #
                        N_str ='N={: 4d}'.format(bool_flag.sum())#   'N=%4d' % bool_flag.sum()
                        axes_abd[i,j].annotate(N_str, xy=(0.1, 0.1), xycoords='axes fraction', fontsize=8)
                #

              
            if j==0:
                axes_abd[i,j].set_ylabel('NEWR')
        axes_abd[i,0].set_xlim(0,10)
        axes_abd[i,1].set_xlim(-10,0)
        axes_abd[i,2].set_xlim(4800,8921)
#            axes_abd[i,j].set_ylim(top=y_sup, bottom=y_inf)
#            axes_abd[i,j].set_xlim(x_inf,x_sup)
            #axes_abd[i,j].set_xscale('log')
#            axes_abd[i,j].set_yscale('symlog')

#            axes_abd[i,j].set_xticks([1,10,20,50,100,200,500,1000,2000,5000])

#    ####

    canvas_abd.draw()
##########################################
def plot_spec(*args):
    #this function shows the spectra and the residuals.

    #clean the axes_c
    for ax in axes_c:
        ax.clear()

    color_stars = params.color_stars
    line_style=['-','--','-','--']

    #set the limits of the wavelentgh window in calibration
    wmin_calib = classes.LineList.wmin_calib
    wmax_calib = classes.LineList.wmax_calib
    wdelta = wmax_calib-wmin_calib
    xmin = wmin_calib-wdelta/5.
    xmax = wmax_calib+wdelta/5.
    #sum up the orig_indexes and the calib_indexes (so that it show also the newly added hyperfine splitting lines)
    indexes_orig_calib = llist.orig_indexes.join(llist.calib_indexes, how='outer')
    #sum up the orig_indexes and the strong_indexes
    indexes_orig_calib_strong = indexes_orig_calib.join(llist.strong_indexes, how='outer')
    indexes_unique = indexes_orig_calib_strong.unique()
    #this is useful in the next loop to set the ylim
    y_lim_resid=[]


    # loop on the list_obs_sp spectra
    for i,ax in enumerate(axes_c[:-1]):
        color_number = i % params.n_spec
        wave_flag = (obj_star_list[i].observed_sp_df.wavelength>=xmin) & (obj_star_list[i].observed_sp_df.wavelength<xmax)
        wave=obj_star_list[i].observed_sp_df.wavelength[wave_flag]
        wave_calib_flag = (wave>=wmin_calib) & (wave<wmax_calib)
        name_star=obj_star_list[i].name_star
        name_obj=obj_star_list[i].name_obj
        flux_obs=obj_star_list[i].observed_sp_df.flux[wave_flag]
        flux_synt_VALD=obj_star_list[i].ini_synthetic_sp_df.flux[wave_flag]
        flux_norm_obs=obj_star_list[i].observed_sp_df.norm[wave_flag]
        flux_synt_calib=obj_star_list[i].synthetic_sp_df.flux[wave_flag]
        flux_residuals=obj_star_list[i].residuals_sp_ds[wave_flag]
        fmax = np.max(flux_norm_obs[wave_flag])
        fmin = np.min(flux_norm_obs[wave_flag])
#        fmax = np.max(flux_synt_VALD[wave_flag])
#        fmin = np.min(flux_synt_VALD[wave_flag])
        ymax = np.max([1.02, fmax+0.02])
        ymin = fmin-0.02
        deltay = ymax-ymin

        #
        # plot the line's center as vertical lines
        for idx in indexes_unique:
            w=llist.calibrated_full_ll_df.loc[idx,('atomic_pars','wavelength')]
            if llist.calibrated_full_ll_df.loc[idx,('atomic_pars','comment')]=='calib':
                ax.plot([w,w],[ymin,ymax], color='grey',linestyle='--')
            elif llist.calibrated_full_ll_df.loc[idx,('atomic_pars','comment')]=='strong':
                ax.plot([w,w],[ymin,ymax], color='lightgrey',linestyle='--')
            elif llist.calibrated_full_ll_df.loc[idx,('atomic_pars','comment')]=='drop':
                ax.plot([w,w],[ymin,ymax], color='lightgrey',linestyle='--')
        # plot grid
        ax.grid(color='lightgrey', which='major', linestyle='--', alpha=0.5)
        #
        ax.text(xmin, fmin, name_star, color=color_stars[color_number])
        # plot constant line y=1
        ax.plot([xmin, xmax], [1, 1], color='grey',linestyle=':')
        #plot obs spec
        ax.plot(wave, flux_obs, color=color_stars[color_number], linestyle='-', linewidth=3, alpha=0.2)
        #plot synt spec with VALD log gf
        ax.plot(wave, flux_synt_VALD, color='grey', linestyle='--', linewidth=3, alpha=0.3)
        #plot obs normalized spec
        ax.plot(wave, flux_norm_obs, color=color_stars[color_number], linestyle='-', linewidth=1)
        #plot synt spec with calibrated log gf
        ax.plot(wave, flux_synt_calib, color='black', linestyle='--', linewidth=1)
        #shade the window under calibration
        ax.fill_between([xmin,wmin_calib], [0], [ymax,ymax], facecolor='lightgrey', alpha=0.6)
        ax.fill_between([wmax_calib, xmax], [0], [ymax,ymax], facecolor='lightgrey', alpha=0.6)

        # ylim
        flux_all=np.concatenate((flux_synt_VALD, flux_norm_obs, flux_synt_calib), axis=0)
#        y_max=np.max(flux_all)
#        y_min=np.min(flux_all)
        ax.set_ylim(top=ymax, bottom=ymin)
        ax.set_xlim(xmin,xmax)
        ax.set_xticks([])
        ax.set_ylabel('flux')
        ###


        #shade the window under calibration
        y_lim_resid.append(flux_residuals[wave_calib_flag].values.tolist())
        #plot in axes_c[5] the residuals
        axes_c[len(axes_c)-1].plot(wave, flux_residuals, color=color_stars[color_number], linestyle='-', linewidth=1)
        axes_c[len(axes_c)-1].set_ylabel('residuals')
        axes_c[len(axes_c)-1].set_xlabel(r'wavelength ($\AA$)')

    y_max=np.max(y_lim_resid)*1.1
    y_min=np.min(y_lim_resid)*1.5
#    axes_c[len(axes_c)-1].fill_between([wmin_calib,wmax_calib], [y_min], [y_max, y_max], facecolor='ghostwhite')
    axes_c[len(axes_c)-1].fill_between([xmin,wmin_calib], [y_min], [y_max, y_max], facecolor='lightgrey', alpha=0.6)
    axes_c[len(axes_c)-1].fill_between([wmax_calib, xmax], [y_min], [y_max, y_max], facecolor='lightgrey', alpha=0.6)
    # plot constant line y=1
    axes_c[len(axes_c)-1].plot([xmin, xmax], [0, 0], color='grey',linestyle=':')
    # limits
    axes_c[len(axes_c)-1].set_ylim(top=y_max, bottom=y_min*1.2)
    axes_c[len(axes_c)-1].set_xlim(wmin_calib-wdelta/5.,wmax_calib+wdelta/5.)


    ####  add the position and  Z of the element lines 
    for i,idx in enumerate(indexes_unique):
        if i % 2 == 0:
            vstep = -15
        else:
            vstep = -30

        w=llist.calibrated_full_ll_df.loc[idx,('atomic_pars','wavelength')]
        if llist.calibrated_full_ll_df.loc[idx,('atomic_pars','comment')]=='calib':
            axes_c[len(axes_c)-1].plot([w,w],[y_min*0.6,y_max], color='grey',linestyle='--')
            axes_c[len(axes_c)-1].annotate(str(llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','atom')]), xy=(llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','wavelength')], y_min*0.6), xytext=(-15,vstep), 
                        textcoords='offset pixels', color='black')
        elif (llist.calibrated_full_ll_df.loc[idx,('atomic_pars','comment')]=='drop') or (llist.calibrated_full_ll_df.loc[idx,('atomic_pars','comment')]=='strong'):
            axes_c[len(axes_c)-1].plot([w,w],[y_min*0.6,y_max], color='lightgrey',linestyle='--')
            axes_c[len(axes_c)-1].annotate(str(llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','atom')]), xy=(llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','wavelength')], y_min*0.6), xytext=(-15,vstep), 
                        textcoords='offset pixels', color='lightgrey')

    ####

    canvas_calib.draw()
##########################################
def show_VdW_lines():
    #show the elements among which one choose one element to show the lines.

    idx_lines = llist.VdW_indexes[VdW_inf.get():VdW_sup.get()]
    #set the intserted_VdW_llist variable        
    VdW_list = classes.df_to_list_of_strings(llist.calibrated_full_ll_df.loc[idx_lines], llist.format_calib_boxes)
    inserted_VdW_llist.set(VdW_list)

##########################################
def plot_lines_previous(*args):
    #this function is associated to the button "previous" in tab3.
    #it plots the previous 3 lines with respect to the present showed lines.

    global counter_lines
    global ele_chosen

    ix=()
    ix = ele_chose_box_tab3.curselection()
    atomic_number = llist.calibrated_elements[ix[0]]

    if ele_chosen!=atomic_number:
        ele_chosen=atomic_number
        counter_lines=min(0,counter_lines)

    idx_flag = (llist.calibrated_full_ll_df.loc[:,('atomic_pars','atom')].astype(int) == atomic_number) &\
               (llist.calibrated_full_ll_df.loc[:,('atomic_pars','comment')] == 'calib') &\
               (llist.calibrated_full_ll_df.loc[:,('atomic_pars','wavelength')] > wave_start_tab3.get())
    #select the lines as function of EW and purity
    idx_flag_2 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'EW')].apply(lambda x: True if (x>ew_minim.get()).any() else False, axis=1)
    idx_flag_3 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'purity')].apply(lambda x: True if (x>purity_tab3.get()).any() else False, axis=1)
    idx_flag[idx_flag==True] = idx_flag_2 & idx_flag_3
    idx_lines = llist.calibrated_full_ll_df[idx_flag].index

    counter_lines = counter_lines - 3
    if counter_lines < 0:
        counter_lines = 0

    inf = counter_lines
    sup = counter_lines + min(len(idx_lines), 3)

    VdW_inf.set(inf)
    VdW_sup.set(sup)

    show_VdW_lines()
##########################################
def plot_lines_next(*args):
    #this function is associated to the button "next" in tab3.
    #it plots the next 3 lines with respect to the present showed lines.

    global counter_lines
    global ele_chosen


    ix=()
    ix = ele_chose_box_tab3.curselection()
    atomic_number = llist.calibrated_elements[ix[0]]

    if ele_chosen!=atomic_number:
        ele_chosen=atomic_number
        counter_lines=min(0,counter_lines)

    idx_flag = (llist.calibrated_full_ll_df.loc[:,('atomic_pars','atom')].astype(int) == atomic_number) &\
               (llist.calibrated_full_ll_df.loc[:,('atomic_pars','comment')] == 'calib') &\
               (llist.calibrated_full_ll_df.loc[:,('atomic_pars','wavelength')] > wave_start_tab3.get())
    #select the lines as function of EW and purity
    idx_flag_2 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'EW')].apply(lambda x: True if (x>ew_minim.get()).any() else False, axis=1)
    idx_flag_3 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'purity')].apply(lambda x: True if (x>purity_tab3.get()).any() else False, axis=1)
    idx_flag[idx_flag==True] = idx_flag_2 & idx_flag_3
    idx_lines = llist.calibrated_full_ll_df[idx_flag].index

    counter_lines = counter_lines + 3
    if counter_lines >= len(idx_lines)-3:
        counter_lines = max(0,len(idx_lines)-3)

    inf = counter_lines
    sup = min(len(idx_lines), counter_lines + 3)

    VdW_inf.set(inf)
    VdW_sup.set(sup)

    show_VdW_lines()
##########################################
def plot_lines_ini(*args):
    #this function is associated to the button "initialize" in tab3.
    
    global counter_lines
    global ele_chosen

    ix=()
    ix = ele_chose_box_tab3.curselection()
    if(len(ix)<1):
        print('choose one element!')
        return
    atomic_number = llist.calibrated_elements[ix[0]]

    if ele_chosen!=atomic_number:
        ele_chosen=atomic_number
        counter_lines=min(0,counter_lines)

    idx_flag = (llist.calibrated_full_ll_df.loc[:,('atomic_pars','atom')].astype(int) == atomic_number) &\
               (llist.calibrated_full_ll_df.loc[:,('atomic_pars','comment')] == 'calib') &\
               (llist.calibrated_full_ll_df.loc[:,('atomic_pars','wavelength')] > wave_start_tab3.get())
    #select the lines as function of EW and purity
    idx_flag_2 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'EW')].apply(lambda x: True if (x>ew_minim.get()).any() else False, axis=1)
    idx_flag_3 = llist.calibrated_full_ll_df[idx_flag].loc[:,(slice(None),'purity')].apply(lambda x: True if (x>purity_tab3.get()).any() else False, axis=1)
    idx_flag[idx_flag==True] = idx_flag_2 & idx_flag_3
    idx_lines = llist.calibrated_full_ll_df[idx_flag].index

    inf = counter_lines
    sup = min(len(idx_lines), counter_lines + 3)

    llist.VdW_indexes = idx_lines
    VdW_inf.set(inf)
    VdW_sup.set(sup)
    show_VdW_lines()

    for ax in axes_l.reshape(-1):
        ax.clear()


##########################################
def plot_lines():
    #this function is associated to the button "plot" in tab3.
    #It plots three lines per time for each spectrum.
    
    idx_lines = llist.VdW_indexes[VdW_inf.get():VdW_sup.get()]

    color_stars = params.color_stars

    for j,idx in enumerate(idx_lines):
        line_number = str(VdW_inf.get() + j + 1)
        w_line = llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','wavelength')]
        atom_str = str(llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','atom')])
        atom_int = llist.calibrated_full_ll_df.loc[idx, ('atomic_pars','atom')].round(0)

        w_sup = w_line + 1.
        w_inf = w_line - 1.
        w_step = w_sup-w_inf
        #prepare the synthetic and observed spectra in the gived wavelength interval
        prepare_synth_obs_spectra(w_inf,w_step)
        #set boolean that select the lines in the wavelength window
        wave_lines_bool = (llist.calibrated_full_ll_df.loc[:,('atomic_pars','wavelength')]>=w_inf) & (llist.calibrated_full_ll_df.loc[:,('atomic_pars','wavelength')]<=w_sup)

        # write on the plot the line number for this element
        text = utils.number_to_name_dict[atom_int] + ' line N.' + line_number + '/' + str(len(llist.VdW_indexes))
        axes_l[0,j].set_title(text)



        for i in range(len(obj_star_list)):
            color_number = i % params.n_spec
            #set boolean that select the spectrum wavelength window
            wave_sp_bool = (obj_star_list[i].observed_sp_df.wavelength>=w_inf) & (obj_star_list[i].observed_sp_df.wavelength<w_sup)
            wave=obj_star_list[i].observed_sp_df.wavelength[wave_sp_bool]
            flux_norm_obs=obj_star_list[i].observed_sp_df.norm[wave_sp_bool]
            flux_synt_calib=obj_star_list[i].synthetic_sp_df.flux[wave_sp_bool]
            purity = llist.calibrated_full_ll_df.loc[idx, (obj_star_list[i].name_obj,'purity')].round(2)

            # plot constant line y=1
            axes_l[i,j].plot([w_inf, w_sup], [1, 1], color='grey',linestyle=':')
            #plot obs normalized spec
            axes_l[i,j].plot(wave, flux_norm_obs, color=color_stars[color_number], linestyle='-', linewidth=1)
            #plot synt spec with calibrated log gf
            axes_l[i,j].plot(wave, flux_synt_calib, color='black', linestyle='--', linewidth=1)
            axes_l[i,j].set_xlim(w_inf, w_sup)
            if i==len(obj_star_list)-1:
                axes_l[i,j].set_xlabel(r'wavelength ($\AA$)')
            if j==0:
                axes_l[i,j].set_ylabel('flux')

            y_inf, y_sup = axes_l[i,j].get_ylim()
            y_delta = y_sup-y_inf

            for k,line_idx in enumerate(llist.calibrated_full_ll_df.index[wave_lines_bool]): 
                w=llist.calibrated_full_ll_df.loc[line_idx,('atomic_pars','wavelength')]


                if llist.calibrated_full_ll_df.loc[line_idx,('atomic_pars','comment')]!='drop' and line_idx!=idx:
                    axes_l[i,j].plot([w,w],[y_inf+y_delta*0.3,y_sup], color='grey',linestyle='--')
                    if i==(len(obj_star_list)-1):
                        axes_l[i,j].annotate(str(llist.calibrated_full_ll_df.loc[line_idx, ('atomic_pars','atom')]), xy=(w, y_inf+y_delta*0.2), color='black', rotation=90, fontsize=8)
                else:
                    axes_l[i,j].plot([w_line,w_line],[y_inf+y_delta*0.3,y_sup], color='red',linestyle='--')

            #annotate the purity
            text = 'purity = ' + str(purity)
            axes_l[i,j].annotate(text, xy=(0.7, 0.1), xycoords='axes fraction', fontsize=8).set_bbox(dict(edgecolor='red', facecolor='white', alpha=1))


            if i!=(len(obj_star_list)-1):
                axes_l[i,j].set_xticks([])
                axes_l[i,j].tick_params(axis='y', direction='in', pad=-20)
            else:
                axes_l[i,j].tick_params(axis='x', rotation=45)
                axes_l[i,j].tick_params(axis='y', direction='in', pad=-20)
                axes_l[i,j].annotate(atom_str, xy=(w_line, y_inf+y_delta*0.2), color='red', rotation=90, fontsize=8)


    ####

    canvas_l.draw()

##########################################
def run_VdW_calib(*args):
    #this function is associated to the "calibrate VdW" button in tab3.
    #It runs an automatic calibration if the Van der Waals parameter of the selected line.
    
    ix=()
    ix = VdW_calib_box.curselection()
    indexes_str = llist.VdW_indexes[VdW_inf.get():VdW_sup.get()]
    indexes = indexes_str[ix[0]]
    wave_line = llist.calibrated_full_ll_df.loc[indexes, ('atomic_pars','wavelength')]
    wave_step = 4.0
    wave_start = wave_line - wave_step/2.0

    #prepare the synthetic and observed spectra in the gived wavelength interval
    prepare_synth_obs_spectra(wave_start,wave_step)

    #minimize
    minim.minimize_VdW_spectra(obj_star_list, llist, indexes)
    #compute the EWs
    minim.compute_equivalent_widths(obj_star_list, llist)
    #compute the NEWRs
    minim.compute_newr(obj_star_list, llist)
    #compute the purity parameter
    minim.compute_purity(obj_star_list, llist)


##########################################
def run_calib():
    #this is the main calibration routine that
    #run after the button "calibrate" in tab1 is pressed.
    #It calibrates the log gfs of the lines visualized in the
    #graphical window by the "plot_spec" function.

    #synthesize the spectra
    minim.synthesize_spectra(obj_star_list)
    #normalize the observed spectra
    minim.normalize_spectra(obj_star_list)
    #minimize
    minim.minimize_spectra(obj_star_list, llist)
    #compute the EWs
    minim.compute_equivalent_widths(obj_star_list, llist)
    #compute the NEWRs
    minim.compute_newr(obj_star_list, llist)
    #compute the purity parameter
    minim.compute_purity(obj_star_list, llist)
    #remove the lines with extreme log gf
    llist.reject_lines()
    #recompute the synthetic spectra, normalize the obs ones, 
    #and compute the residuals after pruning the llist
    synth_norm_residuals()
    #update the ListBox for calibrated line list
    inserted_calib_llist.set(llist.calibrating_interval_list)
    #compute the strong lines, if any
    strong.compute_strong_lines(obj_star_list)
    #clean the axes_c
    for ax in axes_c:
        ax.clear()
    #plot figure
    plot_spec()

##########################################
def initialize_wave_interval(sel_ll_bool):
    #this function initialize many values needed to visualize and calibrate
    #the lines in the wavelength interval considered.
    
    global counter

    #these three lines are used when the "pruned list" flag in tab1 is set.
    if(sel_ll_bool.get()):
        llist.hide_weak_lines(pruning_value.get())        
        llist.calibrated_full_ll_df[('atomic_pars','comment')] = llist.swap_df[('atomic_pars','comment')]

    #prepare the synthetic and observed spectra in the given wavelength interval
    prepare_synth_obs_spectra(wave_start.get(),wave_step.get())
    #set the name of the working line list file name
    classes.Star.ll_name = llist.ll_name
    #set the llist_hyperfine intervals
    llist_hyperfine.set_hyperf_intervals()
    # set hfs interval
    llist_hfs_sneden.set_hyperf_intervals()
    # set luke interval
    llist_luke.set_luke_intervals()
    # set the variables for line list ListBox
    inserted_calib_llist.set(llist.calibrating_interval_list)
    inserted_orig_llist.set(llist.orig_interval_list)
    inserted_hyperf_line.set(llist_hyperfine.orig_interval_list)
    inserted_hfs_sneden_ll.set(llist_hfs_sneden.orig_interval_list)
    inserted_luke_llist.set(llist_luke.orig_interval_list)
    #synthesize the initial spectra (the ones with original, non-calibrated log gfs)
    llist.write_synt_orig_ll()
    minim.synthesize_orig_spectra(obj_star_list)
    #clean the axes_c
    for ax in axes_c:
        ax.clear()
    #initialize guess wavelength
    wave_guess.set(wave_start.get())
##########################################
def prepare_synth_obs_spectra(wmin_synt, w_step):
    #this is a subroutine used in functions where spectrum synthesis is needed,
    #such as "run_calib", "recompute_newr", "plot_lines"
    #"run_VdW_calib", and others.



    #set the name of the working line list file name
    classes.Star.ll_name = llist.ll_name
    #set the line lists class variables
    llist.set_wave_limits(obj_star_list[0].full_observed_sp_df.wavelength, wmin_synt, w_step)
    #set the llist intervals
    llist.set_llist_intervals()
    #set the star class variables
    classes.Star.wave_max = classes.LineList.wmax_synt
    classes.Star.wave_min = classes.LineList.wmin_synt
    #prepare the current observed spectrum interval
    minim.set_observed_spectra_intervals(obj_star_list)
    #prepare the synthetic spectra, normalize the obs ones and compute the residuals
    synth_norm_residuals()
#########################################
def synth_norm_residuals():
    #this function normalize the observed spectra and compute the residuals.
    #It is used in"run_calib" and "prepare_synth_obs_spectra" functions.

    #synthesize the spectra
    minim.synthesize_spectra(obj_star_list)
    #normalize the observed spectra
    minim.normalize_spectra(obj_star_list)
    #compute residuals
    minim.residuals_spectra(obj_star_list)
##########################################
def prev_interval():
    #this function is associated to the button "previous" in tab1.
    #It select the previous wavelength interval with respect to the present showed.
    global counter
    counter = max(0,abs(df_w_intervals['wave_start']-wave_start.get()).idxmin() - 1)
    wave_start.set(df_w_intervals.wave_start.iloc[counter])
    wave_step.set(df_w_intervals.wave_step.iloc[counter])

###########################################
def next_interval():
    #this function is associated to the button "next" in tab1.
    #It select the next wavelength interval with respect to the present showed.
    global counter
    counter = max(0,abs(df_w_intervals['wave_start']-wave_start.get()).idxmin() + 1)
    wave_start.set(df_w_intervals.wave_start.iloc[counter])
    wave_step.set(df_w_intervals.wave_step.iloc[counter])
###########################################
def clean_up():
    #before to quit, remove files used for synthesis/ew computation/ML
    #that are now useless
    command = 'rm *.rsp ML_guess* ML.model ews_star* linelist_star* synt_star*'
    os.system(command)
###########################################
def write_and_quit():
    #This function is associated to the button "write and quit" in tab1.
    #It writes the line list in "llist_table" file and quit
    llist.write_calibrated_full_ll(params.calibrated_file)
    #remove useless files
    clean_up()
    root.quit()

###########################################
def just_quit():
    #This function is associated to the button "quit" in tab1.
    #remove useless files
    clean_up()
    #It does nothing else than quit.
    root.quit()

###############################

### THE PROGRAM STARTS HERE ###
# move into the working directory
os.chdir(params.working_dir)
###############################
#instantiate the VALD list and spectra objects
obj_star_list, llist = utils.instantiate_spectra_lists()
#instantiate additional llists
llist_hyperfine, llist_hfs_sneden, llist_luke = utils.instantiate_additional_lists()
#we use the function define_w_intervals to break the line list into pieces on which
#the calibration will be performed. Define in value_wave_ini and value_wave_end the
#wavelength values covered by both spectra and line list. 
value_wave_ini = max(obj_star_list[0].full_observed_sp_df.wavelength.iloc[0], llist.full_ll_df.atomic_pars.wavelength.iloc[0])
value_wave_end = min(obj_star_list[0].full_observed_sp_df.wavelength.iloc[-1], llist.full_ll_df.atomic_pars.wavelength.iloc[-1])
df_w_intervals = utils.define_w_intervals(llist, value_wave_ini, value_wave_end)
#counter is a global variable that keep trace of the present calibrating interval selected from df_w_intervals
counter=0
#counter_lines is a global variable that keep trace of the lines showed in the plot of tab3 (function plot_lines)
counter_lines=0
#wavelenght step between the intervals in  df_w_intervals. To date, this is fixed at 1.2A in utils.define_w_intervals 
value_wave_step = df_w_intervals.wave_step.iloc[0]
###############################

### here starts the building of the graphical interface

##########################################
root = Tk()
root.geometry("1200x1200")
root.title("line list calibrator")

tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)   # first tab, which would get widgets gridded into it
tab2 = ttk.Frame(tab_control)   # second tab
tab3 = ttk.Frame(tab_control)   # third tab

tab_control.add(tab1, text='calibrate loggfs')
tab_control.add(tab2, text='set abundances')
tab_control.add(tab3, text='plot lines')

#########  set the tab1 window
lf1 = ttk.Labelframe(tab1, padding=8, text='Wavelength window')
lf1.grid(column=0, row=0, sticky=(W,N,S), columnspan=5)
#
lf11 = ttk.Labelframe(tab1, text='line list pruning')
lf11.grid(column=1, row=0, columnspan=1, rowspan=1, sticky=(W, N, S))
#
lf20 = ttk.Labelframe(tab1, padding=8, text='add VALD lines')
lf20.grid(column=0, row=1, rowspan=2, sticky=W)
#
lf201 = ttk.Labelframe(tab1, padding=8, text='add SPECTRUM lines')
lf201.grid(column=0, row=3, rowspan=2, sticky=W)
#
lf21 = ttk.Labelframe(tab1, padding=8, text='add Kurucz hfs splitting lines')
lf21.grid(column=0, row=5, rowspan=2, sticky=W)
#
lf22 = ttk.Labelframe(tab1, padding=8, text='add lines manually')
lf22.grid(column=0, row=8, sticky=W)
#
lf23 = ttk.Labelframe(tab1, padding=8, text='add hfs by Sneden website')
lf23.grid(column=0, row=7, rowspan=1, sticky=W)
#
lf3 = ttk.Labelframe(tab1, padding=8, text='calibrating line list')
lf3.grid(column=1, row=2, rowspan=7, padx=10)
#
lf4 = ttk.Labelframe(tab1, padding=8, text='guess the line with ML algorithm')
lf4.grid(column=0, row=9, sticky=W)
#

# set the windows that hold the figures
figu_c_frame=Toplevel(tab1)
figu_l_frame=Toplevel(tab3)
figu_abd_frame=Toplevel(tab2)
######################################            
#define figure log gfs calibration object (tab1)
figu_c = Figure(figsize=(12,10))
h_ratio = [1 for i in range(len(obj_star_list))]
h_ratio.append(2)
gs = gridspec.GridSpec(len(obj_star_list)+1, 1, hspace=0.0, height_ratios=h_ratio)
axes_c = [figu_c.add_subplot(gs[i]) for i in range(len(obj_star_list)+1)]

figu_c.subplots_adjust(hspace=0.0)
canvas_calib = FigureCanvasTkAgg(figu_c, master=figu_c_frame)
canvas_calib.get_tk_widget().grid(column=2, row=0, columnspan=8, sticky=(NE))

#define figure lines object (tab3)
figu_l, axes_l = plt.subplots(ncols=3, nrows=len(obj_star_list),
                    figsize=(12,10), constrained_layout=False)
figu_l.subplots_adjust(hspace=0.0, wspace=0.03)
canvas_l = FigureCanvasTkAgg(figu_l, master=figu_l_frame)
canvas_l.get_tk_widget().grid(column=2, row=0, columnspan=8)

#define figure newrs object (tab2)
figu_abd, axes_abd = plt.subplots(ncols=4, nrows=len(obj_star_list), sharex=False, sharey=True,
                figsize=(12,10), constrained_layout=False)
figu_abd.subplots_adjust(hspace=0, wspace=0)
canvas_abd = FigureCanvasTkAgg(figu_abd, master=figu_abd_frame)
canvas_abd.get_tk_widget().grid(column=2, row=0, columnspan=8, sticky=(NE))
###############################################

### set buttons/labels/entries in lf1
#w_ini
ttk.Label(lf1, text="wavelength:").grid(column=0, row=0)
wave_start=DoubleVar(value=value_wave_ini)
wave_start_entry = ttk.Entry(lf1, width=7, textvariable=wave_start)
wave_start_entry.grid(column=1, row=0, pady=5)
#wave_step
ttk.Label(lf1, text="step:").grid(column=2, row=0)
wave_step=DoubleVar(value=value_wave_step)
wave_step_entry = ttk.Entry(lf1, width=4, textvariable=wave_step)
wave_step_entry.grid(column=3, row=0, pady=5)
#button to set the previous wavelenght window
prev_button = ttk.Button(lf1, text="previous", command=prev_interval)
prev_button.grid(column=1, row=1, padx=5, pady=5, sticky=E)
#button to set the next wavelenght window
ttk.Button(lf1, text="next", command=next_interval).grid(column=3, row=1, padx=5, pady=5, sticky=W)
#separator
ttk.Separator(lf1, orient=VERTICAL).grid(column=4, row=0, rowspan=2, padx=20, sticky=(NS))
#radio button for selected llist
sel_ll_bool = BooleanVar(value=False)
sel_ll_radiobutton = ttk.Radiobutton(lf1, text='full llist', variable=sel_ll_bool, value=False).grid(column=5, row=0, padx=5, pady=20, sticky=(W))
byhand_radiobutton = ttk.Radiobutton(lf1, text='pruned llist', variable=sel_ll_bool, value=True).grid(column=5, row=1, padx=5, pady=20, sticky=(W))
ttk.Label(lf1, text="value:").grid(column=6, row=1)
pruning_value=DoubleVar(value=0.02)
pruning_entry = ttk.Entry(lf1, width=4, textvariable=pruning_value).grid(column=7, row=1)
#separator
ttk.Separator(lf1, orient=VERTICAL).grid(column=8, row=0, rowspan=2, padx=20, sticky=(NS))
#button to write the present linelist and initialize
ini_button = ttk.Button(lf1, text="initialize", command=lambda : initialize_wave_interval(sel_ll_bool))
ini_button.grid(column=9, row=0, padx=20, pady=5, ipady=20, ipadx=20)
#button to plot spectra
ttk.Button(lf1, text="plot spectra", command=plot_spec).grid(column=9, row=1, padx=20, pady=5, ipady=20, ipadx=20, sticky=N)



### set buttons/labels/entries in lf20

#prepare the listbox for the original VALD line list
inserted_orig_llist = StringVar()
llist_orig_box = Listbox(lf20, listvariable=inserted_orig_llist, height=5, width=55, font=('TkFixedFont',10))
llist_orig_box.grid(column=0, row=0, sticky=W)
### prepare the button to select the VALD lines
add_orig_line_button = ttk.Button(lf20, text='add', command= lambda: add_to_linelist(llist_orig_box,llist), default='active')
add_orig_line_button.grid(column=1, row=0, padx=5,pady=5)

### set buttons/labels/entries in lf201

#prepare the listbox for the original SPECTRUM line list
inserted_luke_llist = StringVar()
llist_luke_box = Listbox(lf201, listvariable=inserted_luke_llist, height=5, width=55, font=('TkFixedFont',10))
llist_luke_box.grid(column=0, row=0, sticky=W)
### prepare the button to select the SPECTRUM lines
add_luke_line_button = ttk.Button(lf201, text='add', command= lambda: add_to_linelist(llist_luke_box,llist_luke), default='active')
add_luke_line_button.grid(column=1, row=0, padx=5,pady=5)

### set buttons/labels/entries in lf21

#prepare the listbox for kurucz hyperfine splitting line list
inserted_hyperf_line = StringVar()
hyperf_box = Listbox(lf21, listvariable=inserted_hyperf_line, height=5, width=55, font=('TkFixedFont',10))
hyperf_box.grid(column=0, row=0, sticky=W)
## prepare the button to select hyperfine splitting lines
add_hyperf_line_button = ttk.Button(lf21, text='add', command= lambda: add_to_linelist(hyperf_box,llist_hyperfine), default='active')
add_hyperf_line_button.grid(column=1, row=0, padx=5,pady=5)
#hyperf_box.bind('<Double-1>', select_line)

### set buttons/labels/entries in lf22 

#prepare the listbox to add a line manually
line_added_manually = StringVar(value='5213.500 90.0 30000 50000 -1.0 1.0 99 MAN')
llist_manual_box = Entry(lf22, textvariable=line_added_manually, width=55, font=('TkFixedFont',10))
llist_manual_box.grid(column=0, row=0, sticky=W) 
## button to move the line to the calibrating list
add_manually_button = ttk.Button(lf22, text='add', command=add_manual_line, default='active')
add_manually_button.grid(column=1, row=0, padx=5,pady=5)

### set buttons/labels/entries in lf23
#prepare the button to upload Sneden hfs file
inserted_hfs_sneden_ll = StringVar()
llist_hfs_sneden_box = Listbox(lf23, listvariable=inserted_hfs_sneden_ll, height=3, width=55, font=('TkFixedFont',10))
llist_hfs_sneden_box.grid(column=0, row=0, rowspan=2, sticky=W)
### prepare the button to select the hfs Sneden lines
add_hfs_sneden_line_button = ttk.Button(lf23, text='add', command= lambda: add_to_linelist(llist_hfs_sneden_box,llist_hfs_sneden), default='active')
add_hfs_sneden_line_button.grid(column=1, row=1, padx=5,pady=5)

### set buttons/labels/entries in lf4

#label and entry to add the wavelength of the guess line
ttk.Label(lf4, text="wavelength:").grid(column=0, row=0, sticky=W)
wave_guess=DoubleVar(value=None)
wave_guess_entry = ttk.Entry(lf4, textvariable=wave_guess, width=10)
wave_guess_entry.grid(column=1, row=0, sticky=W)
### prepare the button to train the ML model
guess_line_button = ttk.Button(lf4, text='train the ML', command=utils.train_guess_ML_model, default='active')
guess_line_button.grid(column=6, row=0, padx=5,pady=5)
### prepare the button to guess the line
guess_line_button = ttk.Button(lf4, text='guess!', command= guess_routine, default='active')
guess_line_button.grid(column=6, row=1, padx=5,pady=5)
#prepare the list box containing the guessed lines
lines_guess= StringVar()
lines_guess_box = Listbox(lf4, listvariable=lines_guess, height=5, width=55, font=('TkFixedFont',10))
lines_guess_box.grid(column=0, row=1, columnspan=5, rowspan=2) 
### prepare the button to add the guessed line
add_guess_line_button = ttk.Button(lf4, text='add', command= add_guess_line, default='active')
add_guess_line_button.grid(column=6, row=2, padx=5,pady=5)

#prepare the listbox for the calibrating line list (lf3)

#calibrating listbox
inserted_calib_llist = StringVar()
llist_calib_box = Listbox(lf3, listvariable=inserted_calib_llist, height=20, width=55, font=('TkFixedFont',10))
llist_calib_box.grid(column=0, columnspan=2, row=0, sticky=(E,W))
#add scrollbar
scrollbar = Scrollbar(lf3, orient="horizontal", command=llist_calib_box.xview)
llist_calib_box.configure(xscrollcommand=scrollbar.set)
scrollbar.grid(column=0, row=1, columnspan=2, sticky=(E,W))
## prepare the button to reject lines in the line list
remove_line_button = ttk.Button(lf3, text='remove lines from line list', command=remove_calib_lines, default='active')
remove_line_button.grid(column=3, row=0, sticky=W)
## prepare the button to move the line from the calibrating list
move_in_line_button = ttk.Button(lf3, text='from line list', command=move_in_line, default='active')
move_in_line_button.grid(column=0, row=2, sticky=W)
## prepare the button to move the line to the calibrating list
move_out_line_button = ttk.Button(lf3, text='to line list', command=move_out_line, default='active')
move_out_line_button.grid(column=1, row=2, sticky=E)
#prepare the listbox to edit the lines
edit_line = StringVar()
llist_edit_box = Entry(lf3, textvariable=edit_line, width=55)
llist_edit_box.grid(column=0, row=3, columnspan=2, sticky=W)

#button to run the calibration
ttk.Button(tab1, text="calibrate", command=run_calib).grid(column=2, row=0, padx=20, pady=20, ipady=20, ipadx=10, sticky=N)

# progressbar
#prog_bar = ttk.Progressbar(lf4, orient=HORIZONTAL, length=100, mode='indeterminate')
#prog_bar.grid(column=9, row=2)

#button to write and quit
ttk.Button(tab1, text="write and quit", command=write_and_quit).grid(column=2, row=9,  padx=20, pady=20, ipady=20, ipadx=10, sticky=S)
#button to quit
ttk.Button(tab1, text="quit", command=just_quit).grid(column=1, row=9,  padx=20, pady=20, ipady=20, ipadx=10, sticky=S)

####
#
#### tab2: newr and abundances ##############################################
#
####

### labelframe
lf1_tab2 = ttk.Labelframe(tab2, text='elements to plot')
lf1_tab2.grid(column=0, row=0, sticky=(NW), columnspan=3)
#entry for the elements
ele_names = StringVar()
ele_chose_box = Listbox(lf1_tab2, listvariable=ele_names, height=50, selectmode=MULTIPLE)
ele_chose_box.grid(column=0, row=2, sticky=(W,E), columnspan=1, rowspan=5)

#button to show elements
ttk.Button(lf1_tab2, text="show elements", command=show_elements).grid(column=0, row=0, padx=20, pady=20, ipady=20, ipadx=10)
#button to plot
ttk.Button(lf1_tab2, text="plot NEWRs", command=plot_newrs).grid(column=1, row=0, padx=20, pady=20, ipady=20, ipadx=10)
#entry box for the purity parameter
ttk.Label(lf1_tab2, text="minimum purity").grid(column=0, row=1, sticky=(W), pady=20)
purity_par = DoubleVar(value=0.0)
purity_box = Entry(lf1_tab2, textvariable=purity_par, width=5)
purity_box.grid(column=1, row=1)

#button for MOOG-like plot
ttk.Button(lf1_tab2, text="plot params check\n (MOOG like)", command=plot_params_check).grid(column=1, row=3, padx=20, pady=20, ipady=20, ipadx=10)

### labelframe
# prepare the spinbox reporting all the elements found in the files star?.abd 
spinbox_abd_dict = {}
lf2_tab2 = ttk.Labelframe(tab2, text='abundances to change')
lf2_tab2.grid(column=4, row=0, sticky=(N,W,E), columnspan=6, padx=20)
for col, star in enumerate(obj_star_list[1:]):
    row= 0
    ttk.Label(lf2_tab2, text=star.name_star).grid(column=col+1, row=row, sticky=(W), padx=20)
    for key, val in star.abd_dict.items():
        row += 1
        ttk.Label(lf2_tab2, text=key).grid(column=0, row=row+1, sticky=(W), padx=20)
        name_spinbox = star.name_star + key  + '_spinbox'
        spinbox_abd_dict[name_spinbox] = abd_spinbox(lf2_tab2, col, row+1, key, val)
#button to write the star?.abd files
ttk.Button(lf2_tab2, text="write the abd files", command=write_abd_atm_files).grid(column=9, row=40, padx=20, pady=20, ipady=20, ipadx=10, sticky=(S))
#button to compute the NEWRs for all the elements
ttk.Button(lf2_tab2, text="recompute NEWRs", command=recompute_newr).grid(column=10, row=40, padx=20, pady=20, ipady=20, ipadx=10, sticky=(S))
####

### tab3 plot lines ############################

####
### labelframe
lf1_tab3 = ttk.Labelframe(tab3, text='element lines to plot')
lf1_tab3.grid(column=1, row=1, sticky=(NW), columnspan=3)
# automatic calibration of the VdW parameter
lf2_tab3 = ttk.Labelframe(tab3, text='calibrate Van der Waals parameter')
lf2_tab3.grid(column=5, row=1, sticky=(NW), columnspan=3)
#entry for the elements
ele_chose_box_tab3 = Listbox(lf1_tab3, listvariable=ele_names, height=55, exportselection=False)
ele_chose_box_tab3.grid(column=1, row=0, sticky=(NW), columnspan=1, rowspan=10)
ele_chosen=0

#button to show elements
ttk.Button(lf1_tab3, text="show elements", command=show_elements).grid(column=2, row=0, padx=20, pady=5, ipady=20, ipadx=10)
#button to initialize
ttk.Button(lf1_tab3, text="initialize", command=plot_lines_ini).grid(column=3, row=0, padx=20, pady=5, ipady=20, ipadx=40)
#button to plot
ttk.Button(lf1_tab3, text="plot", command=plot_lines).grid(column=4, row=0, padx=20, pady=5, ipady=20, ipadx=20)
#button to plot next lines
ttk.Button(lf1_tab3, text="next", command=plot_lines_next).grid(column=3, row=1)
#button to plot next lines
ttk.Button(lf1_tab3, text="previous", command=plot_lines_previous).grid(column=2, row=1)
#entry box for w_ini
ttk.Label(lf1_tab3, text="wavelength:").grid(column=2, row=2, pady=5)
wave_start_tab3=DoubleVar(value=4800.0)
wave_start_entry_tab3 = ttk.Entry(lf1_tab3, width=7, textvariable=wave_start_tab3)
wave_start_entry_tab3.grid(column=3, row=2, pady=5)
#entry box for the minimum EW
ttk.Label(lf1_tab3, text="minimum EW (mA)").grid(column=2, row=3, pady=5)
ew_minim = DoubleVar(value=0.0)
ew_box = Entry(lf1_tab3, textvariable=ew_minim, width=5)
ew_box.grid(column=3, row=3, pady=5)
#entry box for the minimum purity
ttk.Label(lf1_tab3, text="minim purity:").grid(column=2, row=4, pady=5)
purity_tab3=DoubleVar()
purity_entry_tab3 = ttk.Entry(lf1_tab3, width=7, textvariable=purity_tab3)
purity_entry_tab3.grid(column=3, row=4, pady=5)


#calibrating VdW listbox
inserted_VdW_llist = StringVar()
VdW_inf = IntVar()
VdW_sup = IntVar()
VdW_calib_box = Listbox(lf2_tab3, listvariable=inserted_VdW_llist, height=5, width=50, font=('TkFixedFont',10), exportselection=False)
VdW_calib_box.grid(column=0, columnspan=2, row=0, rowspan=3, sticky=(E,W))

#button to run
ttk.Button(lf2_tab3, text="calibrate VdW", command=run_VdW_calib).grid(column=3, row=1, padx=20, pady=20, ipady=20, ipadx=10)
undo_VdW_button = ttk.Button(lf2_tab3, text='undo VdW', command=undo_VdW, default='active').grid(column=3, row=2, padx=20, pady=20, ipady=20, ipadx=10)


####################################
#set the tabs
tab_control.grid(column=0, row=0)


root.mainloop()
