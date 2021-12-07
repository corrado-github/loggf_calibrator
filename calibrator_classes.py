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


import sys, os, re
import numpy as np
import pandas as pd
import os.path
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


#################################
def upload_hfs_sneden1(file):

    if re.search('Co',file):
#        formatter={'wavelength': '{:9.4f}','lEv': '{:7.0f}','hEv': '{:7.0f}',
#                   'loggf': '{:>6.2f}','Waals': '{:.3f}','transition': '{:4s}','source': '{:4s}'}
        formatter={'wavelength': 'f8','lEv': 'f8','hEv': 'f8',
                   'loggf': 'f8'}
        column_names = ['wavelength', 'hEv', 'lEv', 'loggf']
        df=pd.read_csv(file, sep='\s+', names=column_names, usecols=[0,1,4,9], dtype=formatter)
        df['atom'] = 27.0
        df['Waals'] = 1.0001
        df['transition'] = '99'
        df['source'] = 'Lawler2015'


    df.atom=df.atom.round(1)
    df.lEv=df.lEv.round(1)
    df.hEv=df.hEv.round(1)
    df.loggf=df.loggf.round(2)
    df.Waals=df.Waals.round(3)
    return df
#################################
def upload_hfs_sneden(list_files):

    list_columns=['wavelength','atom','lEv','hEv','loggf','Waals','transition','source']
    list_df = []

    for i, file in enumerate(list_files):
        list_df.append(pd.read_csv(file, sep='\s+', header=0))

        if re.search('Co',file):
            source_str = 'Lawler2015'
            list_df[i]['atom'] = 27.0
        elif re.search('Mn',file):
            source_str = 'DenHartog2011'
            list_df[i]['atom'] = list_df[i]['J_up'].apply(lambda x: 25.0 if bool(x%1) else 25.1) 
        elif re.search('V',file):
            source_str = 'Lawler2015'
            list_df[i]['atom'] = 23.0

        list_df[i]['Waals'] = 1.0001
        list_df[i]['transition'] = '99'
        list_df[i]['source'] = source_str

    df = pd.concat(list_df, axis=0, ignore_index=True, sort=False)
    df.sort_values(by='wavelength', ascending=True, inplace=True)

    df.atom=df.atom.round(1)
    df.lEv=df.lEv.round(1)
    df.hEv=df.hEv.round(1)
    df.loggf=df.loggf.round(2)
    df.Waals=df.Waals.round(3)
    return df[list_columns]

#################################
def upload_hyperfine(file):
    formatter_str={'wavelength': '{:8.3f}'.format,'atom': '{:4.1f}'.format,'lEv': '{:7.0f}'.format,'hEv': '{:7.0f}'.format,
            'loggf': '{:>6.2f}'.format,'Waals': '{:.3f}'.format,'transition': '{:4s}'.format,'source': '{:4s}'.format}
    formatter={'wavelength': '{:9.4f}','atom': '{:4.1f}','lEv': '{:7.0f}','hEv': '{:7.0f}',
            'loggf': '{:>6.2f}','Waals': '{:.3f}','transition': '{:4s}','source': '{:4s}'}

    width_list=[10,7,6,12,5,11,12,5,11,6,6,6,4,2,2,3,6,3,6,5,5,1,1,1,1,1,1,1,3,5,5,6]
    list_names=['wavelength','atom','lEv','hEv','loggf','Waals','transition','source']

    df=pd.read_fwf(file, widths=width_list, usecols=[0,1,2,3,6,11,12,13], dtype=formatter)
    df.columns=['wavelength','loggf','atom','lEv','hEv','Waals','source','transition']
    df=df[list_names]
    df.transition=99
    df.wavelength=(df.wavelength*10.).round(3)
    df.atom=df.atom // 1 + (df.atom % 1)*10.
    df.atom=df.atom.round(1)
    df.lEv=df.lEv.round(1)
    df.hEv=df.hEv.round(1)
    df.loggf=df.loggf.round(2)
    df.Waals=1.0001
    df.Waals=df.Waals.round(3)
    df.transition = df.transition.astype(str)
    df.source = df.source.astype(str)

    return df
##########################################
#it seems that in the Kurucz hfs line lis,t some lines the lEv and hEv are reversed
#here I fix them
def fix_kurucz_levels(df):


    boole = df[('atomic_pars','hEv')] < df[('atomic_pars','lEv')]
    df1 = df[boole]['atomic_pars'].copy()
    df1['lEv_swap'] = df1['lEv']

    cols = ['lEv','hEv']
    df1['lEv_swap'] = df1[cols].apply(lambda x: x.min(), axis=1)
    df1['hEv'] = df1[cols].apply(lambda x: x.max(), axis=1)
    df1['lEv'] = df1['lEv_swap']
    df1.drop(columns='lEv_swap', inplace=True)
    df1 = pd.concat([df1], axis=1, keys=['atomic_pars'])
    df.update(df1)

    return df
#################################
def upload_llist(file):

    list_names=['wavelength','atom','lEv','hEv','loggf','Waals','transition','source']
    list_dtype=['f8','f8','f8','f8','f8','f8','S','S']
    dict_dtype=dict(zip(list_names,list_dtype))
    df = pd.read_csv(file, names=list_names, sep='\s+', dtype=dict_dtype)

    df.wavelength = df.wavelength.round(3)
    df.lEv = df.lEv.round(1)
    df.hEv = df.hEv.round(1)
    df.loggf = df.loggf.round(2)
    df.Waals = 1.0001
    df.Waals = df.Waals.round(3)

#    #remove the H lines because they are already hard coded in SPECTRUM 
#    H_lines_to_drop_index = df.index[(df.atom==1.0) | (df.atom==2.0)]
#    df.drop(index=H_lines_to_drop_index, inplace=True)
#    #remove the NaI lines because they are already hard coded in SPECTRUM
#    Na_lines_to_drop_index = df.index[(df.atom==11.0) & ((df.wavelength==5889.951) | (df.wavelength==5895.924))]
#    df.drop(index=Na_lines_to_drop_index, inplace=True)

    return df

#################################
def allowed_species(file):
    df_atom = pd.read_csv(file, delimiter='\s+', header=0)
    list = []
    for i,row in df_atom.iterrows():
        for charge in np.arange(row.maxcharge+1):
            specie = np.round(row.code + charge/10., 1) 
            list.append(specie)
    return set(list)
##################################
def df_to_list_of_strings(df,formatter):

    lista=[]

    for index,row in df.iterrows():
        line = ''

        for item, value in row.iteritems():
            if item[1] in list(formatter):
                line = line + ' ' + formatter[item[1]](value)
        lista.append(line)

    return lista
##################################
def ds_to_list_of_strings(ds,formatter):

    line = ''
    for item, value in ds.iteritems():
        if item[1] in list(formatter):
            line = line + ' ' + formatter[item[1]](value)

    return line
#################################
def compute_line_integration_interval(wavelength, sigma, ew):
    #it compute the inf and sup of a wavelength interval
    #given EW and sigma of a line, it computes the inf and sup wavelength
    #where the flux of the line is equal to 1-alpha
    #The profile is assumed to be Gaussian
    alpha = 0.5# 0.5 means half-maximum
    mean_gauss_norm = norm.pdf(0)

    mean_gauss_height = norm.pdf(0, scale=sigma)*ew
 
    alpha_norm = mean_gauss_norm*(alpha/mean_gauss_height)

#    print('sigma, ew',sigma, ew)
#    print('norm', mean_gauss_norm)
#    print('ews', mean_gauss_height, alpha, alpha_norm)
#    print('mean_gauss_norm,mean_gauss_height,ratio,alpha_norm',mean_gauss_norm,mean_gauss_height,ratio,alpha_norm)

    #take at minimum 0.03A as half_width to have a reasonable average over the line
    half_width = np.max([0.03,sigma * np.sqrt(-2.*np.log(min(0.99,alpha_norm*np.sqrt(2*np.pi))))])
    wave_low_lim = wavelength - half_width
    wave_high_lim = wavelength + half_width
#    print('interval', half_width*sigma, wave_low_lim, wave_high_lim)

    return wave_low_lim, wave_high_lim

#################################
class LineList():

    ####
    light_speed = 299792.458
    wmin_synt = None
    wmax_synt = None
    wmin_calib = None
    wmax_calib = None
    extra_width = 5
    ll_name = 'linelist'

    par_names=['wavelength','atom','lEv','hEv','loggf','Waals','transition','source']
    par_orig_names=['wavelength','atom','lEv','hEv','loggf_orig','Waals_orig','transition','source']
    par_full_names=['wavelength','atom','lEv','hEv','loggf','loggf_orig','Waals','Waals_orig','transition','source','comment']


    format={'wavelength': '{:8.3f}'.format,'atom': '{:>4.1f}'.format,'lEv': '{:7.0f}'.format,'hEv': '{:7.0f}'.format,
            'loggf': '{:>6.2f}'.format,'Waals': '{:.1f}'.format,'transition': '{:s}'.format,'source': '{:s}'.format}
    format_orig_boxes={'wavelength': '{:>8.3f}'.format,'atom': '{:>5.1f}'.format,'lEv': '{:>7.0f}'.format,'hEv': '{:>7.0f}'.format,
            'loggf_orig': '{:>+6.2f}'.format,'Waals_orig': '{:>5.1f}'.format,'transition': '{:s}'.format,'source': '{:>5s}'.format}
    format_calib_boxes={'wavelength': '{:>8.3f}'.format,'atom': '{:>5.1f}'.format,'lEv': '{:>7.0f}'.format,'hEv': '{:>7.0f}'.format,
            'loggf': '{:>+6.2f}'.format,'Waals': '{:>5.1f}'.format,'transition': '{:s}'.format,'source': '{:>5s}'.format,
            'EW': '{:>5.1f}'.format}


    format_dict={'wavelength': 'float64','atom': 'float64', 'lEv': 'float64','hEv': 'float64',
            'loggf': 'float64','Waals': 'float64','transition': 'str','source': 'str'}


    synt_interval_list = None
    calibrating_interval_list = None
    VdW_interval_list = None
    calibrated_full_ll_df = None
    editing_line = None
    index_to_edit = None


    synt_indexes = []
    calib_indexes = []
    strong_indexes = []
    VdW_indexes = None

    def __init__(self):
        self.full_ll_df=None
        self.orig_interval_df = None
        self.orig_interval_list = None
        self.orig_indexes = []
        self.swap_df = pd.DataFrame(None, columns=pd.MultiIndex.from_product([['atomic_pars'],['comment','selected']]))
        self.calibrated_elements = []
        self.n_manual_index = -1
        self.par_stars = ['EW', 'strength', 'NEWR', 'purity', 'strength_synt', 'strength_obs', 'abs_synt_flux', 'strength_contr']#, 'deriv1_diff']

    def upload_ll(self,file, file_calibr, n_spectra, stdatom_file):
        self.full_ll_df = upload_llist(file)
        self.full_ll_df.rename(lambda x: '_'.join((str(x),'VALD')), axis='index', inplace=True)
        #prepare the columns with original values and given order
        self.full_ll_df['loggf_orig'] = self.full_ll_df['loggf']
        self.full_ll_df['Waals_orig'] = self.full_ll_df['Waals']
        #add comment column
        self.full_ll_df['comment'] = 'none'
        #order the columns
        self.full_ll_df = self.full_ll_df[self.par_full_names]

        # remove the ionization not supported by SPECTRUM
        set_species = allowed_species(stdatom_file)
#        self.full_ll_df['comment'] = self.full_ll_df['atom'].apply(lambda x: 'drop' if x not in set_species else 'none')
        species_to_drop = [x for x in self.full_ll_df['atom'].unique() if x not in set_species]
        for item in species_to_drop:
            boole = (self.full_ll_df['atom'] == item)
            self.full_ll_df.drop(index=self.full_ll_df.index[boole],inplace=True)
        #drop the lines with loggf>3.0 or loggf<-10.0
        lines_to_drop_index = self.full_ll_df.index[(self.full_ll_df.loggf<-10.0) | (self.full_ll_df.loggf>3.0)]
        self.full_ll_df.drop(index=lines_to_drop_index,inplace=True)
        #comment the H lines because they are already hard coded in SPECTRUM 
        H_lines_index = self.full_ll_df.index[(self.full_ll_df.atom==1.0) | (self.full_ll_df.atom==2.0)]
        self.full_ll_df.loc[H_lines_index, 'comment'] = 'strong'
        #comment the NaI lines because they are already hard coded in SPECTRUM
        Na_lines_index = self.full_ll_df.index[(self.full_ll_df.atom==11.0) & ((self.full_ll_df.wavelength==5889.951) | (self.full_ll_df.wavelength==5895.924))]
        self.full_ll_df.loc[Na_lines_index, 'comment'] = 'strong'


        #create second level column multiindexes
        self.full_ll_df = pd.concat([self.full_ll_df], axis=1, keys=['atomic_pars'])


        #
        file = file_calibr

        if os.path.isfile(file):
            LineList.calibrated_full_ll_df  = pd.read_csv(file, header=[0,1], index_col=0)#dtype = format_dict
            LineList.calibrated_full_ll_df.loc[:, ('atomic_pars','transition')] = LineList.calibrated_full_ll_df.loc[:,('atomic_pars','transition')].astype(str)         
            #set the n_manual_index if any 
            manual_indexes_list = [int(x.strip('manual_')) for x in LineList.calibrated_full_ll_df.index if re.search('manual',x)]
            if len(manual_indexes_list)>0:
                self.n_manual_index = max(manual_indexes_list)

            #check if all the columns in self.par_stars are present. If not, add them.
            star_label_list = LineList.calibrated_full_ll_df.columns.get_level_values(0).unique()[1:].tolist()

            for star_label in star_label_list:
                star_cols_list = LineList.calibrated_full_ll_df[star_label].columns.unique().tolist()
                for col in self.par_stars:
                    if col not in star_cols_list:
                        LineList.calibrated_full_ll_df[(star_label,col)] = np.nan
                    
        else:
            LineList.calibrated_full_ll_df = self.full_ll_df.copy()
            list_star = []
            #set the EWs columns
            for i in range(n_spectra):
                list_star.append('star' + str(i))

            cols_index = pd.MultiIndex.from_product([list_star,self.par_stars])
            df_temp = pd.DataFrame(index=LineList.calibrated_full_ll_df.index, columns=cols_index)
            LineList.calibrated_full_ll_df = pd.concat([LineList.calibrated_full_ll_df,df_temp], axis=1, join='outer', sort=False)

    def upload_hyperfine_ll(self,file, n_spectra):
        self.full_ll_df = upload_hyperfine(file)
        self.full_ll_df.rename(lambda x: '_'.join((str(x),'hyperf')), axis='index', inplace=True)
        #prepare the columns with original values and given order
        self.full_ll_df['loggf_orig'] = self.full_ll_df['loggf']
        self.full_ll_df['Waals_orig'] = self.full_ll_df['Waals']
        #reorder the columns
        cols = self.full_ll_df.columns.tolist()
        cols = [cols[x] for x in [0,1,2,3,4,8,5,9,6,7]]
        self.full_ll_df = self.full_ll_df[cols]
        #add comment column
        self.full_ll_df['comment'] = 'none'
        self.full_ll_df = pd.concat([self.full_ll_df], axis=1, keys=['atomic_pars'])

        list_star = []
        #set the EWs columns
        for i in range(n_spectra):
            list_star.append('star' + str(i))

        cols_index = pd.MultiIndex.from_product([list_star,self.par_stars])
        df_temp = pd.DataFrame(index=self.full_ll_df.index, columns=cols_index)
        self.full_ll_df = pd.concat([self.full_ll_df,df_temp], axis=1, join='outer', sort=False)

    def upload_hfs_sneden_ll(self,list_files, n_spectra):
        self.full_ll_df = upload_hfs_sneden(list_files)
        self.full_ll_df.rename(lambda x: '_'.join((str(x),'hfs_sneden')), axis='index', inplace=True)
        #prepare the columns with original values and given order
        self.full_ll_df['loggf_orig'] = self.full_ll_df['loggf']
        self.full_ll_df['Waals_orig'] = self.full_ll_df['Waals']
        self.full_ll_df['comment'] = 'none'
        #reorder the columns
#        cols = self.full_ll_df.columns.tolist()
#        cols = [cols[x] for x in [0,1,2,3,4,8,5,9,6,7]]
        self.full_ll_df = self.full_ll_df[self.par_full_names]
        self.full_ll_df = pd.concat([self.full_ll_df], axis=1, keys=['atomic_pars'])

        list_star = []
        #set the EWs columns
        for i in range(n_spectra):
            list_star.append('star' + str(i))

        cols_index = pd.MultiIndex.from_product([list_star,self.par_stars])
        df_temp = pd.DataFrame(index=self.full_ll_df.index, columns=cols_index)
        self.full_ll_df = pd.concat([self.full_ll_df,df_temp], axis=1, join='outer', sort=False)

    def upload_luke_ll(self,file, n_spectra):
        self.full_ll_df = upload_llist(file)
        self.full_ll_df.rename(lambda x: '_'.join((str(x),'luke')), axis='index', inplace=True)
        #prepare the columns with original values and given order
        self.full_ll_df['loggf_orig'] = self.full_ll_df['loggf']
        self.full_ll_df['Waals_orig'] = self.full_ll_df['Waals']
        #reorder the columns
        cols = self.full_ll_df.columns.tolist()
        cols = [cols[x] for x in [0,1,2,3,4,8,5,9,6,7]]
        self.full_ll_df = self.full_ll_df[cols]
        #add comment column
        self.full_ll_df['comment'] = 'none'
        self.full_ll_df = pd.concat([self.full_ll_df], axis=1, keys=['atomic_pars'])

        list_star = []
        #set the EWs columns
        for i in range(n_spectra):
            list_star.append('star' + str(i))

        cols_index = pd.MultiIndex.from_product([list_star,self.par_stars])
        df_temp = pd.DataFrame(index=self.full_ll_df.index, columns=cols_index)
        self.full_ll_df = pd.concat([self.full_ll_df,df_temp], axis=1, join='outer', sort=False)


    def set_wave_limits(self, wave_obs_sp_ds, w_ini, step):
        #in case the observed spectrum wavelength is not sequential, the following choice
        #should give the correct wmin and wmax
        LineList.wmin_synt=max(wave_obs_sp_ds[wave_obs_sp_ds>=w_ini-LineList.extra_width].iloc[0], w_ini-LineList.extra_width)
        LineList.wmax_synt=min(wave_obs_sp_ds[wave_obs_sp_ds<=w_ini+step+LineList.extra_width].iloc[-1], w_ini+step+LineList.extra_width)
        LineList.wmin_calib=max(wave_obs_sp_ds[wave_obs_sp_ds>=w_ini].iloc[0], w_ini)
        LineList.wmax_calib=min(wave_obs_sp_ds[wave_obs_sp_ds<=w_ini+step].iloc[-1], w_ini+step)

    def set_llist_intervals(self):
        #set the boolean to select the lines inside the wavelength window and that are not strong
        boole = (LineList.calibrated_full_ll_df.atomic_pars.wavelength>=LineList.wmin_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.wavelength<LineList.wmax_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.comment != 'strong')
        #set the orig interval
        self.orig_indexes = LineList.calibrated_full_ll_df.index[boole]
#        (LineList.calibrated_full_ll_df.atomic_pars.wavelength>=LineList.wmin_calib) & (LineList.calibrated_full_ll_df.atomic_pars.wavelength<LineList.wmax_calib)]
        self.orig_interval_df = LineList.calibrated_full_ll_df.loc[self.orig_indexes]
        # set the *_list lists
        self.orig_interval_list = df_to_list_of_strings(self.orig_interval_df.loc[self.orig_indexes], self.format_orig_boxes)
        #set the boolean to select the lines inside the synt wavelength window and that are not drop nor strong
        boole = (LineList.calibrated_full_ll_df.atomic_pars.wavelength>=LineList.wmin_synt) &\
               (LineList.calibrated_full_ll_df.atomic_pars.wavelength<LineList.wmax_synt) &\
               (LineList.calibrated_full_ll_df.atomic_pars.comment != 'drop') &\
               (LineList.calibrated_full_ll_df.atomic_pars.comment != 'strong')
        #set the synt interval
        LineList.synt_indexes = LineList.calibrated_full_ll_df.index[boole]

        #set the boolean to select the lines inside the calib wavelength window and that are not drop nor strong
        boole = (LineList.calibrated_full_ll_df.atomic_pars.wavelength>=LineList.wmin_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.wavelength<LineList.wmax_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.comment != 'drop') &\
               (LineList.calibrated_full_ll_df.atomic_pars.comment != 'strong')
        #set the calib interval (which is a subsample of synt_interval_df)
        LineList.calib_indexes = LineList.calibrated_full_ll_df.index[boole]
        # set the calibrating lines
        LineList.calibrated_full_ll_df.loc[LineList.calib_indexes, ('atomic_pars','comment')] = 'calib'

        # set the *_list lists
        LineList.synt_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.synt_indexes,('atomic_pars',self.par_names)], self.format)
        LineList.calibrating_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes], self.format_calib_boxes)

        #set the boolean to select the strong lines inside the calib wavelength window
        boole = (LineList.calibrated_full_ll_df.atomic_pars.wavelength>=LineList.wmin_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.wavelength<LineList.wmax_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.comment == 'strong')
        #set the strong indexes
        if boole.sum()>0:
            LineList.strong_indexes = LineList.calibrated_full_ll_df.index[boole]
        else:
            LineList.strong_indexes = []

        #write the 'linelist'
        self.write_synt_ll()

    def set_hyperf_intervals(self):
        orig_indexes = self.full_ll_df.index[(self.full_ll_df.loc[:,('atomic_pars','wavelength')] >= LineList.wmin_calib) & (self.full_ll_df.loc[:,('atomic_pars','wavelength')] < LineList.wmax_calib)]
        self.orig_interval_df = self.full_ll_df.loc[orig_indexes]
        #fix the kurucz levels
        self.orig_interval_df = fix_kurucz_levels(self.orig_interval_df)
        #
        self.orig_interval_list = df_to_list_of_strings(self.orig_interval_df.loc[orig_indexes], self.format_orig_boxes)

    def set_luke_intervals(self):
        orig_indexes = self.full_ll_df.index[(self.full_ll_df.loc[:,('atomic_pars','wavelength')] >= LineList.wmin_calib) & (self.full_ll_df.loc[:,('atomic_pars','wavelength')] < LineList.wmax_calib)]
        self.orig_interval_df = self.full_ll_df.loc[orig_indexes]
        #
        self.orig_interval_list = df_to_list_of_strings(self.orig_interval_df.loc[orig_indexes], self.format_orig_boxes)

    def set_swap_df(self):
        if (self.swap_df.shape[0] == LineList.calibrated_full_ll_df.shape[0]):
            self.swap_df.update(LineList.calibrated_full_ll_df.loc[:,('atomic_pars','comment')])
#            self.swap_df[('atomic_pars','selected')] = self.swap_df[('atomic_pars','comment')]
        else:
            self.swap_df = self.swap_df.combine_first(LineList.calibrated_full_ll_df.loc[:,('atomic_pars','comment')].to_frame())
#            self.swap_df[('atomic_pars','selected')] = self.swap_df[('atomic_pars','comment')]

    def write_synt_orig_ll(self):
        df = LineList.calibrated_full_ll_df.atomic_pars.reindex(self.par_orig_names, axis=1)
        with open(self.ll_name, 'w') as outfile:
            #here I use synt_indexes (instead of orig_indexes) so that the original VALD log gfs
            #are used in the whole plotted interval.
            df.loc[self.synt_indexes].to_string(outfile,formatters=LineList.format, index=False, header=False)

    def write_synt_ll(self):
        with open(self.ll_name, 'w') as outfile:
            LineList.calibrated_full_ll_df.loc[LineList.synt_indexes,('atomic_pars',self.par_names)].to_string(outfile,formatters=LineList.format, index=False, header=False)

    def write_calibrating_ll(self):
        with open(self.ll_name, 'w') as outfile:
            LineList.calibrated_full_ll_df.loc[LineList.calib_indexes,('atomic_pars',self.par_names)].to_string(outfile,formatters=LineList.format, index=False, header=False)

    def reject_lines(self):
        bool_condition = (LineList.calibrated_full_ll_df.loc[LineList.calib_indexes,('atomic_pars', 'loggf')] <= -10.0) &\
                         (LineList.calibrated_full_ll_df.loc[LineList.calib_indexes,('atomic_pars', 'loggf')] >= 3.0)

        #prepare the indexes of the lines to drop
        indexes_to_drop = LineList.calib_indexes[bool_condition].values.tolist()

        # set the dropping lines
        LineList.calibrated_full_ll_df.loc[indexes_to_drop, ('atomic_pars','comment')] = 'drop'
        # drop the indexes from the calib_indexes and synt_indexes
        LineList.calib_indexes = pd.Index([i for i in LineList.calib_indexes if i not in indexes_to_drop])
        LineList.synt_indexes = pd.Index([i for i in LineList.synt_indexes if i not in indexes_to_drop])

        #set the calibrating interval list
        LineList.calibrating_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes], self.format_calib_boxes)
        #write the 'linelist'

        self.write_synt_ll()

    def hide_weak_lines(self, limit):
        boole = (LineList.calibrated_full_ll_df.atomic_pars.wavelength>=LineList.wmin_calib) &\
               (LineList.calibrated_full_ll_df.atomic_pars.wavelength<LineList.wmax_calib)
        X_df = LineList.calibrated_full_ll_df[boole].copy()
        #set the boolean that identify only the lines that has "strength_contr" smaller than a give value 
        cols = X_df.columns.get_level_values(0).unique()[1:].tolist()
        bool_condition_drop = X_df.loc[:,(cols, 'strength_contr')] < limit
#       #prepare the indexes of the lines
        indexes_to_drop = bool_condition_drop[bool_condition_drop.all(axis=1)].index
        #set the original comments
        LineList.calibrated_full_ll_df.loc[X_df.index,('atomic_pars','comment')] = self.swap_df.loc[X_df.index,('atomic_pars','comment')]
        #set to drop the "indexes_to_drop"
        LineList.calibrated_full_ll_df.loc[indexes_to_drop,('atomic_pars','comment')] = 'drop'

    def show_elements(self):
        bool_condition = (LineList.calibrated_full_ll_df.loc[:,('atomic_pars','comment')] == 'calib')
        df = LineList.calibrated_full_ll_df[bool_condition]
        #prepare the calibrated_elements list (to use for the NEWR plot)
        self.calibrated_elements = sorted(df.loc[:,('atomic_pars','atom')].apply(lambda x: int(x)).unique())

    def remove_lines_update(self, ix):
        indexes_to_drop = [LineList.calib_indexes[i] for i in list(ix)]
        LineList.calib_indexes = pd.Index([ i for i in LineList.calib_indexes if i not in indexes_to_drop])
        LineList.synt_indexes = pd.Index([ i for i in LineList.synt_indexes if i not in indexes_to_drop])
        LineList.calibrated_full_ll_df.loc[indexes_to_drop, ('atomic_pars','comment')] = 'drop'

        #set the synt and calibrating intervals
        LineList.calibrating_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes], self.format_calib_boxes)
        #write the 'linelist'
        self.write_synt_ll()

    def add_lines_update(self, ix):
        rows_to_add = self.orig_interval_df.iloc[list(ix)]
        # add the rows to the synt_interval_df
        for index, row in rows_to_add.iterrows():
            idx = pd.Index([index])
            #check if the line is not in LineList.calib_indexes
            #if so, add it
            if index not in LineList.calib_indexes:
                LineList.calib_indexes = LineList.calib_indexes.append(idx)
                LineList.synt_indexes = LineList.synt_indexes.append(idx)
                #check if the line is already in the calibrated_full_ll_df
                if index not in LineList.calibrated_full_ll_df.index:
                    LineList.calibrated_full_ll_df = LineList.calibrated_full_ll_df.append([row])
                    LineList.calibrated_full_ll_df.loc[idx, ('atomic_pars','comment')] = 'calib'
                else:
                    LineList.calibrated_full_ll_df.loc[idx, ('atomic_pars','comment')] = 'calib'
                    #reset the original loggf and Waals
                    LineList.calibrated_full_ll_df.loc[idx, ('atomic_pars','loggf')] = LineList.calibrated_full_ll_df.loc[idx, ('atomic_pars','loggf_orig')]
                    LineList.calibrated_full_ll_df.loc[idx, ('atomic_pars','Waals')] = LineList.calibrated_full_ll_df.loc[idx, ('atomic_pars','Waals_orig')]

        #sort by wavelength the calibrated_full_ll_df and the calib_indexes 
        LineList.calibrated_full_ll_df.sort_values(by=('atomic_pars','wavelength'), ascending=True, inplace=True)
        flag_sort = list(np.argsort(np.asarray(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes,('atomic_pars','wavelength')])))
        LineList.calib_indexes = pd.Index([LineList.calib_indexes[i] for i in flag_sort])
        #sort by wavelength the synt_indexes 
        flag_sort = list(np.argsort(np.asarray(LineList.calibrated_full_ll_df.loc[LineList.synt_indexes,('atomic_pars','wavelength')])))
        LineList.synt_indexes = pd.Index([LineList.synt_indexes[i] for i in flag_sort])

        #set the calibrating interval list
        LineList.calibrating_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes], self.format_calib_boxes)
        #write the 'linelist'
        self.write_synt_ll()

    def move_in_line(self, idx):
        LineList.index_to_edit = idx
        LineList.editing_line = ds_to_list_of_strings(LineList.calibrated_full_ll_df.loc[idx], self.format)

    def move_out_line(self, line_str):
        line_df = pd.DataFrame([line_str.split()], index=[LineList.index_to_edit], columns=self.par_names)
        line_df = line_df.astype(LineList.format_dict)
        line_df = pd.concat([line_df], axis=1, keys=['atomic_pars'])
        LineList.calibrated_full_ll_df.update(line_df)
        #set the calibrating interval list
        LineList.calibrating_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes,('atomic_pars',self.par_names)], self.format_calib_boxes)
        #write the 'linelist'
        self.write_synt_ll()
        #empty the 
        LineList.index_to_edit = None

    def add_manually_line(self, line_str):
        self.n_manual_index = self.n_manual_index +1
        manual_index = pd.Index(['manual_' + str(self.n_manual_index)])
        line_df = pd.DataFrame([line_str.split()], index=manual_index, columns=self.par_names)
        line_df = line_df.astype(LineList.format_dict)
        line_df['loggf_orig'] = line_df['loggf']
        line_df['Waals_orig'] = line_df['Waals']
        line_df['comment'] = 'calib'
        line_df = pd.concat([line_df], axis=1, keys=['atomic_pars'])


        LineList.calib_indexes = LineList.calib_indexes.append(manual_index)
        LineList.synt_indexes = LineList.synt_indexes.append(manual_index)

        cols=LineList.calibrated_full_ll_df.columns
        LineList.calibrated_full_ll_df = pd.concat([LineList.calibrated_full_ll_df,line_df], axis=0, join='outer', sort=False)
        LineList.calibrated_full_ll_df = LineList.calibrated_full_ll_df[cols]

        #sort by wavelength the calibrated_full_ll_df and the calib_indexes 
        LineList.calibrated_full_ll_df.sort_values(by=('atomic_pars','wavelength'), ascending=True, inplace=True)
        flag_sort = list(np.argsort(np.asarray(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes,('atomic_pars','wavelength')])))
        LineList.calib_indexes = pd.Index([LineList.calib_indexes[i] for i in flag_sort])
        #sort by wavelength the synt_indexes 
        flag_sort = list(np.argsort(np.asarray(LineList.calibrated_full_ll_df.loc[LineList.synt_indexes,('atomic_pars','wavelength')])))
        LineList.synt_indexes = pd.Index([LineList.synt_indexes[i] for i in flag_sort])

        #set the calibrating interval list
        LineList.calibrating_interval_list = df_to_list_of_strings(LineList.calibrated_full_ll_df.loc[LineList.calib_indexes], self.format_calib_boxes)
        #write the 'linelist'
        self.write_synt_ll()



    def write_calibrated_full_ll(self, name_file):
        # select the calibrated lines only
        boole = LineList.calibrated_full_ll_df.loc[:,('atomic_pars', 'comment')] == 'calib'
        index_to_write = LineList.calibrated_full_ll_df.index[boole]
        LineList.calibrated_full_ll_df.loc[index_to_write,('atomic_pars',self.par_names)].to_csv(name_file, header=True, index=False)
        # don't write the hyperfine splitting lines that were drop
        indexes = LineList.calibrated_full_ll_df.index.values.tolist()
        idx_to_keep = [x for x in indexes if re.search('VALD',x) or LineList.calibrated_full_ll_df.loc[x,('atomic_pars', 'comment')] != 'drop']
        df = LineList.calibrated_full_ll_df.loc[idx_to_keep]
        #write the full table
        df.to_csv(name_file, header=True, index=True)        
###################################
# 
#####################################
def write_rsp_ew(llist, abd_dict, name_batch,model,ll_name,wmin,wmax,output,microt, mh):
    import calibrator_utils as utils
    
    ll = llist.calibrated_full_ll_df['atomic_pars'].loc[llist.calib_indexes,llist.par_names].copy()

    for key, value in abd_dict.items():
        boole = ll.atom.astype(int) == utils.atomic_number_dict[key]
        ll.loc[boole,'loggf'] = ll.loc[boole,'loggf'] - (mh - value)
#    ll['loggf'] = ll.apply(lambda x: x.loggf + abd_dict[utils.number_to_name_dict[int(x.atom)]], axis=1)

    with open(ll_name, 'w') as outfile:
        ll.to_string(outfile,formatters=llist.format, index=False, header=False)

    file=open(name_batch,"w")
    file.write('%s\n' % (model))
    file.write('%s\n' % (ll_name))
    file.write('%7.2f,%7.2f\n' % (wmin,wmax))
    file.write('%s\n' % (output))
    file.write('%s\n' % ('stdatom.dat'))
    file.write('%s\n' % (microt))
    file.close()
###################################
def write_rsp_synt(name_batch,model,ll_name,output,abd_file,microt,wmin,wmax,step):

    file=open(name_batch,"w")
    file.write('%s\n' % (model))
    file.write('%s\n' % (ll_name))
    file.write('%s\n' % (abd_file))
    file.write('%s\n' % (output))
    file.write('%s\n' % (microt))
    file.write('%7.2f,%7.2f\n' % (wmin,wmax))
    file.write('%s\n' % (step))
    file.close()
#################################
def upload_spectrum(file):

    list_names=['wavelength','flux']
    list_dtype=['f8','f8']
    dict_dtype=dict(zip(list_names,list_dtype))
    df=pd.read_csv(file, names=list_names, sep='\s+', dtype=dict_dtype, header=None)
    return df

##################################
def upload_ews(file):
    list_names=['wavelength','atom','EW', 'strength']
    list_dtype=['f8','f8','f8','f8']
    dict_dtype=dict(zip(list_names,list_dtype))
    return pd.read_csv(file, names=list_names, sep='\s+', dtype=dict_dtype)
##################################
def cut_spectrum(df, wmin, wmax):
    df1 = df[(df.wavelength>=wmin) & (df.wavelength<wmax)]
    return df1.reset_index(drop=True)
##################################
def normalize_spectrum(df_obs_spec,df_synt_spec):
    obs_flux = np.asarray(df_obs_spec.flux)
    synt_flux = np.asarray(df_synt_spec.flux)
    resid=np.subtract(obs_flux,synt_flux)
    continuum=[]
    radius=500

#    df_obs_spec['norm'] = df_obs_spec['flux']
#    return df_obs_spec
    

    for i,line in enumerate(obs_flux):
        isup=min(i+radius,len(obs_flux))
        iinf=max(i-radius,0)
        f_sp = obs_flux[iinf:isup]
        f_model = synt_flux[iinf:isup]
        f_resid = resid[iinf:isup] - np.mean(resid[iinf:isup])
        sig_r = np.std(f_resid)
        mask = np.abs(f_resid) < 2.*sig_r
        avg_f=np.mean(f_sp[mask])
        avg_m=np.mean(f_model[mask])

        continuum.append(1.+(avg_f-avg_m))

    df_obs_spec['norm'] = pd.Series(np.divide(obs_flux,np.array(continuum)))
    return df_obs_spec

####################################
class Star():
    '''define a star as object'''

    light_speed = 299792.458
    wave_max = None
    wave_min = None
    wave_max_strong = None
    wave_min_strong = None
    wave_ext = 2.0
    ll_name = None

    def __init__(self, llist):
        self.llist = llist
        self.name_obj = None
        self.name_star = None
        self.file_obs_sp = None
        self.file_abd_atm = None
        self.file_abd = None
        self.file_atm_model = None
        self.file_synt_out = None
        self.file_ews_out = None
        self.microt = None
        self.macrot = None
        self.sigma_disp = None
        self.pix_step = None
        self.abd_dict = None
        self.mh = None
        #define a linelist for the EW computation. 
        #It will be different for each star (because the software 'lines' 
        #does not support other abundances than the nominal metallicity).
        self.ll_name_ew = None
        #define the command to use to synthesize the spectrum
        self.file_batch_synt = None
        self.synt_command = None
        #define the command to add macroturbulence to the spectrum
        self.file_macturb = None
        self.macturb_command = None
        #define the command to degrade the spectrum
        self.file_degraded = None
        self.degrade_command = None
        #define the command to compute the EWs
        self.file_batch_ews = None
        self.ews_command = None
        #define a dataframe for the synthetic and observed spectrum
        self.ini_synthetic_sp_df = None
        self.synthetic_sp_df = None
        self.synth_strong_sp_df = None
        self.full_observed_sp_df = None
        self.observed_sp_df = None
        self.residuals_sp_ds = None
        
    def synthesize(self, output, pos):
        #this method synthesizes the spectrum and normalize the observed spectrum
        wmin_ext=max(Star.wave_min-Star.wave_ext, self.llist.wmin_synt)
        wmax_ext=min(Star.wave_max+Star.wave_ext, self.llist.wmax_synt)
        # write the parameter file for SPECTRUM
        write_rsp_synt(self.file_batch_synt,self.file_atm_model,self.llist.ll_name,self.file_synt_out,self.file_abd_atm,self.microt,wmin_ext,wmax_ext,self.pix_step)        
        # run SPECTRUM
        os.system(self.synt_command)
        os.system(self.macturb_command)
        #prepare the degrade command
        wave_mean = np.mean([wmin_ext,wmax_ext])
        sigma_dispersion = np.round(self.sigma_disp(self.name_star,wave_mean),3)
        degrade_command = 'smooth2 ' + self.file_macturb + ' ' + self.file_degraded + ' ' + self.pix_step + ' ' + str(sigma_dispersion) + ' ' + self.pix_step + '\n'
        os.system(degrade_command)
        #cut the synthetic spectrum to the right wavelength window
        df=upload_spectrum(self.file_degraded)
        df1=cut_spectrum(df,Star.wave_min,Star.wave_max)
        output.put((pos,df1))

    def synthesize_strong(self, output, pos):
        #this method synthesizes the spectrum and normalize the observed spectrum
        wmin_ext = Star.wave_min_strong + Star.wave_ext
        wmax_ext = Star.wave_max_strong + Star.wave_ext
        # write the parameter file for SPECTRUM
        write_rsp_synt(self.file_batch_synt,self.file_atm_model,self.llist.ll_name,self.file_synt_out,self.file_abd_atm,self.microt,wmin_ext,wmax_ext,self.pix_step)        
        # run SPECTRUM
        os.system(self.synt_command)
        #cut the synthetic spectrum to the right wavelength window
        df=upload_spectrum(self.file_synt_out)
        df1=cut_spectrum(df,Star.wave_min_strong,Star.wave_max_strong)
        output.put((pos,df1))

    def upload_obs_spectrum(self):
        #upload the observed spectrum
        self.full_observed_sp_df=upload_spectrum(self.file_obs_sp)
        #set the weights
        self.full_observed_sp_df['weights'] = 1.0
        self.full_observed_sp_df['weights'].where(self.full_observed_sp_df['flux']>0.01, 0.0, inplace=True)

    def set_obs_sp_interval(self, output, pos):
        self.observed_sp_df=cut_spectrum(self.full_observed_sp_df,Star.wave_min,Star.wave_max)
        output.put((pos,self.observed_sp_df))

    def normalize(self, output, pos):
        #normalize the observed spectrum
        df_norm = normalize_spectrum(self.observed_sp_df,self.synthetic_sp_df)
        #df_norm = self.observed_sp_df
        #df_norm['norm'] = df_norm['flux']
        output.put((pos,df_norm))

    def integrate_absorbed_flux(self, wmin, wmax, spec_wave, spec_flux):
        #integrate the flux absorbed in a spectrum over the wavelenght interval wmin, wmax
        
        #select the lines in the wavelength window
        boole=(spec_wave>=wmin) & (spec_wave<=wmax)
        #integrated flux absorbed by the spectral line on the observed spectrum
        integrated_flux = (((1.0 - spec_flux[boole]).clip(lower=0.0))*float(self.pix_step)).sum()*1000. #value in mA    
        return np.max([0.001,integrated_flux]) #min 0.001 to avoid division by zero

    def compute_strength(self, spec_wave, spec_flux, wave):
        wave_idx = np.abs(spec_wave-wave).idxmin()
        return 1.-spec_flux[wave_idx]

    def compute_newr(self, output, pos):
        newr_list = []
        strength_synt_list = []
        strength_obs_list = []
        integrated_synt_list = []
#        deriv1_diff_list = []

        #create a list of indexes that contains indexes of calibrating and strong lines (if any) together
        calib_strong_indexes = self.llist.calib_indexes.union(self.llist.strong_indexes)

        for index, row in self.llist.calibrated_full_ll_df.loc[calib_strong_indexes].iterrows():

            name_star = self.name_obj
            ew = row.loc[(self.name_obj,'EW')]/1000.
            wave = row.loc[('atomic_pars','wavelength')]
            sigma_dispersion = self.sigma_disp(self.name_star,wave)
            sigma = np.sqrt(float(sigma_dispersion)**2 + (float(self.macrot)*wave/self.light_speed)**2 + (float(self.microt)*wave/self.light_speed)**2)
            #delta defined as function of EW. This is very similar to the
            #way adopted in Boeche&Grebel 2017.
            #delta = lambda x: 0.05 if x<50 else (0.3 if x>=300 else 0.001*x)
            #wmin=row.loc[('atomic_pars','wavelength')]-delta(ew)
            #wmax=row.loc[('atomic_pars','wavelength')]+delta(ew)
            #
            #define the limits over which we integrate the flux. Different from Boeche&Grebel.
            #call the function to compute wave_low_lim and wave_high_lim
            wmin, wmax = compute_line_integration_interval(row.atomic_pars.wavelength, sigma, ew)
            #integrated flux absorbed by the spectral line on the observed spectrum
            integrated_obs_flux = self.integrate_absorbed_flux(wmin, wmax, self.observed_sp_df.wavelength, self.observed_sp_df.norm)
            #integrated flux absorbed over the spectral line on the synthetic spectrum (in mA)
            integrated_synt_flux = self.integrate_absorbed_flux(wmin, wmax, self.synthetic_sp_df.wavelength, self.synthetic_sp_df.flux)
#            print(integrated_obs_flux,integrated_synt_flux,integrated_obs_flux-integrated_synt_flux)
            newr = np.log10(1.0 + (integrated_synt_flux - integrated_obs_flux)/integrated_obs_flux) #negative means synthetic line strength is smaller than observed
            newr_list.append(newr)
            #now annotate the synt and observed flux at the center of the line
            strength_synt_list.append(self.compute_strength(self.synthetic_sp_df.wavelength, self.synthetic_sp_df.flux, row.loc[('atomic_pars','wavelength')]))
            strength_obs_list.append(self.compute_strength(self.observed_sp_df.wavelength, self.observed_sp_df.norm, row.loc[('atomic_pars','wavelength')]))
            integrated_synt_list.append(integrated_synt_flux)
            
#            #now compute the difference (obs-synt) of the deriv1 around the center of the line
#            wmin=row.loc[('atomic_pars','wavelength')]-0.05
#            wmax=row.loc[('atomic_pars','wavelength')]+0.05
#            boole=(self.synthetic_sp_df.wavelength>=wmin) & (self.synthetic_sp_df.wavelength<=wmax)
#            deriv1_diff = self.observed_sp_df.flux[boole].diff() - self.synthetic_sp_df.flux[boole].diff()
#            deriv1_diff_list.append(deriv1_diff.abs().sum())
#        df= pd.DataFrame({'NEWR': newr_list, 'strength_synt': strength_synt_list, 'strength_obs': strength_obs_list,
#                          'deriv1_diff': deriv1_diff_list}, index=self.llist.calib_indexes)
        df= pd.DataFrame({'NEWR': newr_list, 'strength_synt': strength_synt_list, 'strength_obs': strength_obs_list, 'abs_synt_flux': integrated_synt_list}, index=calib_strong_indexes)
        df = pd.concat([df], axis=1, keys=[self.name_obj])
        output.put((pos, df))

    def compute_residuals(self, output, pos):
        self.residuals_sp_ds = self.observed_sp_df.norm - self.synthetic_sp_df.flux
        resid_weighted = pd.Series(np.multiply(self.residuals_sp_ds, self.observed_sp_df.weights))
        output.put((pos, resid_weighted))
    
    def integrate_residuals(self,wave):
        #comput the sigma profile
        sigma_dispersion = self.sigma_disp(self.name_star,wave)
        sigma = np.sqrt(float(sigma_dispersion)**2 + (float(self.macrot)*wave/self.light_speed)**2)# + (float(self.microt)*wave/self.light_speed)**2)

        wmin = wave-sigma*2.0
        wmax = wave+sigma*2.0
        boole = (self.synthetic_sp_df.wavelength>=wmin) & (self.synthetic_sp_df.wavelength<=wmax)
        flux_to_integrate = np.clip(-self.residuals_sp_ds[boole].values, a_min=0.0, a_max=None)
        return np.sum(flux_to_integrate*float(self.pix_step))*1000./0.95

    def compute_ews(self, output, pos):
        #prepare the parameters file needed by the software 'lines' 
        write_rsp_ew(self.llist,self.abd_dict,self.file_batch_ews,self.file_atm_model,self.ll_name_ew,Star.wave_min,Star.wave_max,self.file_ews_out,self.microt, self.mh)
        # run the software 'lines' that computes the EWs
        os.system(self.ews_command)
        df=upload_ews(self.file_ews_out)
        #set the strength value
        df.loc[:,'strength'] = df.loc[:,'strength'].apply(lambda x: np.round(1.-x,3))
        df.loc[:,'EW'] = df.loc[:,'EW'].apply(lambda x: np.max([0.001,x]))
        df = pd.concat([df], axis=1, keys=[self.name_obj])
        #compute the 'NEWR' over each line
        output.put((pos, df))
