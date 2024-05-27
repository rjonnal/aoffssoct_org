import numpy as np
from matplotlib import pyplot as plt
import sys,os
import data_source as ds
import org_models as om
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
from setup import *
import pandas as pd

print(plt.rcParams['font.size'])
#sys.exit()

image_width_in = 4.75
fig = plt.figure(figsize=(image_width_in,4),dpi=100)

def transfer_function_components(freq,A0,A1,taua,taub):
    t1 = 1j*A0/(2*np.pi*freq)
    t2 = A1/(taua*(2*1j*np.pi*freq/taua+1))
    t3 = A1/(taub*(2*1j*np.pi*freq/taub+1))
    return t1,t2,t3

def transfer_function(freq,A0,A1,taua,taub):
    t1,t2,t3 = transfer_function_components(freq,A0,A1,taua,taub)
    tf = t1 + t2 - t3
    return tf

def cutoff(freq,y):
    lfreq = np.log10(np.abs(freq))
    idx1 = np.where(lfreq<-.7)[0]
    idx2 = np.where(lfreq>0.5)[0]


    x1 = lfreq[idx1]
    y1 = y[idx1]
    x2 = lfreq[idx2]
    y2 = y[idx2]

    p1 = np.polyfit(x1,y1,1)
    p2 = np.polyfit(x2,y2,1)

    # intersection
    x = (p2[1]-p1[1])/(p1[0]-p2[0])
    y = p1[0]*x+p1[1]
    return 10**x,y,p2[0]

df = pd.read_csv('fitted_parameters.csv')
#df = df[df['subject']=='S1']
df = df[df['bleaching_pct']>=2]
print(df)

freq = np.arange(1e-1,30,5e-3).astype(complex)
axes = fig.subplot_mosaic([['a','b'],['c','d']])
axdict = {'S1':['a','c'],'S2':['b','d']}
tf_fullmin = np.inf
tf_fullmax = -np.inf
tf_t2min = np.inf
tf_t2max = -np.inf
colors = {2:(0.8,0.8,0.0),4:(0.4,0.6,0.0),8:(0.5,0.0,0.0),16:(0.8,120.0/255.0,0.0)}
for idx,row in df.iterrows():
    t1,t2,t3 = transfer_function_components(freq,row['A_0'],row['A_1'],row['tau_a'],row['tau_b'])
    bleaching = int(row['bleaching_pct'])
    color = colors[bleaching]
    
    tf_full = t1+t2-t3 
    tf_full = np.abs(tf_full)
    tf_full = tf_full**2
    tf_full_db = 10*np.log10(tf_full)
    tf_fullmin = min(tf_fullmin,np.min(tf_full_db))
    tf_fullmax = max(tf_fullmax,np.max(tf_full_db))

    tf_t2 = t2
    tf_t2 = np.abs(tf_t2).astype(float)
    tf_t2 = tf_t2**2
    tf_t2_db = 10*np.log10(tf_t2)
    tf_t2min = min(tf_t2min,np.min(tf_t2_db))
    tf_t2max = max(tf_t2max,np.max(tf_t2_db))
    xcut,ycut,dbperdecade = cutoff(freq,tf_t2_db)
    print('%0.1f, %0.1f'%(dbperdecade,xcut))
    axes[axdict[row['subject']][0]].semilogx(freq,tf_full_db,color=color,label='%d%%'%row['bleaching_pct'])
    
    axes[axdict[row['subject']][1]].semilogx(freq,tf_t2_db,color=color,label='%d%%'%row['bleaching_pct'])
    axes[axdict[row['subject']][1]].semilogx(xcut,ycut,marker='x',color=color,markersize=2)
    
for subject,panel in zip([1,2],['a','b']):
    #axes[panel].legend()
    axes[panel].set_ylim((tf_fullmin-3,tf_fullmax+3))
    #axes[panel].set_xlabel('frequency (Hz)')
    axes[panel].set_title('subject %d'%subject)
    axes[panel].text(np.min(freq),tf_fullmax+3,'(%s)'%panel,ha='left',va='bottom')
    
for subject,panel in zip([1,2],['c','d']):
    #axes[panel].legend()
    axes[panel].set_ylim((tf_t2min-3,tf_t2max+3))
    #axes[panel].set_xlabel('frequency (Hz)')
    #axes[panel].set_title('subject %d'%subject)
    axes[panel].text(np.min(freq),tf_t2max+3,'(%s)'%panel,ha='left',va='bottom')
    
axes['b'].set_yticks([])
axes['d'].set_yticks([])
axes['a'].set_ylabel('power (dB)')
axes['c'].set_ylabel('power (dB)')
fig.supxlabel('frequency (Hz)')

figsave(fig,'power_spectrum.png')
figsave(fig,'power_spectrum.pdf')

plt.show()
