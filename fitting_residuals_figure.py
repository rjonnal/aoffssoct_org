import numpy as np
from matplotlib import pyplot as plt
import sys,os
import data_source as ds
import org_models as om
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
from setup import *


def add_subplot_labels(axes):
    for label, ax in axes.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        t = ax.text(xlim[0], ylim[1], '(%s)'%label,verticalalignment='top',bbox=dict(pad=3.0,facecolor='1.0',edgecolor='none'))
        t.set_bbox(dict(facecolor='w', alpha=0.0, edgecolor='w'))

bounds_dict = {'A_0':(-100,0),'A_1':(0,1000),'tau_a':(0,50),'tau_b':(0,2),'t_0':(0,50e-3)}
cyan_bounds_dict = {'A_0':(50,150),'A_1':(25,75),'tau_a':(0,20),'tau_b':(0,2)}

r_1sec = om.Red('1sec',bounds_dict)
g_1sec = om.Green('1sec',bounds_dict)
b_1sec = om.Blue('1sec',bounds_dict)
m_1sec = om.Magenta('1sec',bounds_dict)
c_1sec = om.Cyan('1sec')



A0min = -100
A0max = 0
A1min = 0
A1max = 1000
tauamin = 0
tauamax = 20
taubmin = 0
taubmax = 0.5


image_width_in = 4.75

single_flash_data = ds.get_single_flash_data()

# fitting parameters:
# magenta |(0, 8.0)|-39.77|213.49|8.80|0.13|
A_0_m = -39.77
A_1_m = 213.49
tau_a_m = 8.8
tau_b_m = 0.13
data_key = (0,8.0)
t,L = single_flash_data[data_key]

valid = np.where(t<=1.0)
t = t[valid]
L = L[valid]

def weighting_function(t):
    out = np.ones(t.shape)
    out[(t<0.1)] = 0.5
    return out

xlim = (-.1,1.0)
xticks = np.arange(0,1,.25)
fig = plt.figure(figsize=(image_width_in,3))
axes = fig.subplot_mosaic([['a','b'],['a','c'],['a','d'],['a','e'],['a','f']])
axes['a'].plot(t,L,'ko',alpha=0.2,markersize=2)
axes['a'].set_xlim(xlim)
axes['a'].set_xticks(xticks)
axes['a'].set_ylabel('$\Delta OPL$ (nm)')
axes['a'].set_xlabel('time (s)')
for idx,(model,panel) in enumerate(zip([m_1sec,g_1sec,b_1sec,r_1sec,c_1sec],['b','c','d','e','f'])):
    model.fit(t,L,data_key,weighting_function=weighting_function)
    fit = model.model(t,*[model.fitted_parameters[data_key][pn] for pn in model.parameter_names])
    if model.color=='m':
        fit = fit + 1
    if model.color=='g':
        fit = fit - 1
    if model.color=='c':
        fit = fit + 1
    if model.color=='r':
        fit = fit - 1
        
    axes['a'].plot(t,fit,color=model.color,linestyle=model.linestyle,linewidth=1)
    
    if not panel in ['a','f']:
        axes[panel].set_xticks([])
    else:
        axes[panel].set_xticks(xticks)
        axes[panel].set_xlabel('time (s)')
    if not panel in ['a']:
        axes[panel].set_yticks([-20,0,20])
        axes[panel].set_yticklabels([])


    err = L-fit
    trms = np.std(err)
    laterms = np.std(err[t>0.5])
    full_colors = {'m':'magenta','g':'green','b':'blue','r':'red','c':'cyan'}
    print(trms,laterms,'$%s$ & %s & \\SI{%0.1f}{\\nano\\meter} & \\SI{%0.1f}{\\nano\\meter} \\\\'%
          (model.tex().replace('\Delta OPL(t) = ',''),
           full_colors[model.color],
           trms,
           laterms))
           
    #print('%s: $%s$, total RMS error = \\SI[%0.1f][\nano\meter], t>0.5 RMS error = \\SI[%0.1f][\nano\meter]'%(model.color,model.tex(),trms,laterms))
        
    axes[panel].plot(t,L-fit,color=model.color,linestyle=model.linestyle,linewidth=1)
    axes[panel].set_ylim([-40,40])
    axes[panel].grid(axis='y')
    axes[panel].set_xlim(xlim)

    tex = model.tex().replace('\Delta OPL(t) = ','')
    #axes['a'].text(.1,50-idx*12,'$%s$'%tex,color=model.color,fontsize=8)
    axes[panel].text(-.1,22,'$%s$'%tex,color=model.color,fontsize=9,va='bottom')

figsave(fig,'residuals_figure.png')
figsave(fig,'residuals_figure.pdf')
plt.show()
