import numpy as np
from matplotlib import pyplot as plt
#import mpl_interactions.ipyplot as iplt
import scipy.io as sio
import scipy.optimize as spo
import scipy.signal as sps
import sys,os,glob
import octoblob.plotting_functions as opf
import org_models as om

sfd = sio.loadmat('data/single_flash_ORGs_S1.mat',squeeze_me=True)
sfr = sio.loadmat('data/single_flash_ORGs_S2.mat',squeeze_me=True)
afd = sio.loadmat('data/adapting_background_S1.mat',squeeze_me=True)
afr = sio.loadmat('data/adapting_background_S2.mat',squeeze_me=True)
tfd = sio.loadmat('data/two_flashes_S1.mat',squeeze_me=True)
tfr = sio.loadmat('data/two_flashes_S2.mat',squeeze_me=True)
srd = sio.loadmat('data/step_response_S1.mat',squeeze_me=True)
srr = sio.loadmat('data/step_response_S2.mat',squeeze_me=True)
    

def fix_nans(t,y):
    if np.sum(np.isnan(y)):
        t_fixed = t[np.where(1-np.isnan(y))]
        y_fixed = y[np.where(1-np.isnan(y))]
        t = t_fixed
        y = y_fixed
    return t,y

# single_flash_data keys are (subject,bleaching), values are (t,y)
subjects = ['S1','S2']

def get_single_flash_data():

    single_flash_data = {}

    for k in sfd.keys():
        if k.find('mean_org')==-1:
            continue
        bleaching = float(k.replace('mean_org_','').replace('_percent','').replace('p','.'))
        if bleaching in [0.5,32,64]:
            t = sfd['time_0p5_32_64_percent']
        else:
            t = sfd['time']

        t = t + 2e-3
        y = sfd[k]
        y = y - np.nanmean(y[np.where(np.logical_and(t<=0.005,t>-.04))])

        t,y = fix_nans(t,y)

        single_flash_data[(0,bleaching)] = [t,y]
        assert len(t)==len(y)


    for k in sfr.keys():
        if k.find('mean_org')==-1:
            continue
        bleaching = float(k.replace('mean_org_','').replace('_percent','').replace('p','.'))
        t = sfr['time']
        y = y - np.nanmean(y[np.where(np.logical_and(t<=0.005,t>-.04))])
        t = t + 2e-3
        y = sfr[k]
        t,y = fix_nans(t,y)

        single_flash_data[(1,bleaching)] = [t,y]
        assert len(t)==len(y)
        
    return single_flash_data
