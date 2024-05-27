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

m_1sec = om.Magenta('1sec')

image_width_in = 4.75

single_flash_data = ds.get_single_flash_data()
# fitting parameters:
# magenta |(0, 8.0)|-39.77|213.49|8.80|0.13|
A_0_m = -39.77
A_1_m = 213.49
tau_a_m = 8.8
tau_b_m = 0.13
t,L = single_flash_data[(0,8.0)]
fit = m_1sec.model(t,A_0_m,A_1_m,tau_a_m,tau_b_m)

fig = plt.figure(figsize=(image_width_in,5.25))
axes = fig.subplot_mosaic([['a','c'],['a','d'],['b','d'],['b','d']])

axes['a'].plot(t,L,'ko',alpha=0.2,markersize=2)
axes['a'].plot(t,fit)
axes['a'].add_patch(Rectangle((-.05,-50), .1, 100, facecolor=(0.8,0.8,0.8), alpha=0.5))
axes['a'].set_xlim((-.2,1.0))
axes['a'].set_xlabel('time (s)')
axes['a'].set_ylabel('$\Delta OPL$ (nm)')
axes['b'].plot(t*1000,L,'ko',alpha=0.3,markersize=2)
axes['b'].plot(t*1000,fit)
#axes['b'].add_patch(Rectangle((-50,-40), 100, 80, facecolor=(0.8,0.8,0.8), alpha=0.5))
axes['b'].set_facecolor((0.9,0.9,0.9))
axes['b'].set_xlim((-50,50))
axes['b'].set_ylim((-45,45))
axes['b'].set_xlabel('time (ms)')
axes['c'].set_xlabel('time (ms)')
axes['d'].set_xlabel('time (s)')
axes['b'].set_ylabel('$\Delta OPL$ (nm)')

axes['c'].set_yticks([])
axes['d'].set_yticks([])
axes['d'].set_xlim((-.2,1.0))

increments = {'A_0':10.0,'A_1':50.0,'tau_a':6.0,'tau_b':0.15}
linestyles = [':','-','--']

for pidx,param in enumerate(m_1sec.parameter_names):
    if pidx==0:
        ax = axes['c']
        tplot = t*1000
    else:
        ax = axes['d']
        tplot = t
    for fidx,factor in enumerate([-1,0,1]):
        tex_param = param
        if param.find('tau')==0:
            tex_param = '\\'+param
        params = [t,A_0_m,A_1_m,tau_a_m,tau_b_m]
        params[pidx+1] = params[pidx+1]+factor*increments[param]
        dat = m_1sec.model(*params)-250*pidx
        if fidx==1:
            ax.text(0,-250*pidx+5,'$%s$'%tex_param,ha='right',va='bottom')
        ax.plot(tplot,dat,linestyle=linestyles[fidx],label='$%s$'%param,color=colors[0])
        if pidx==0:
            ax.set_xlim((-10,50))
            ax.set_ylim((-50,50))
add_subplot_labels(axes)
#axes['c'].add_patch(Rectangle((-10,-45), 55, 90, facecolor=(0.8,0.8,0.8), alpha=0.5))
axes['c'].set_facecolor((0.9,0.9,0.9))

figsave(fig,'concept_v2.png')
figsave(fig,'concept_v2.pdf')
#fig.savefig('concept_v2.png',dpi=600)
plt.show()
print(axes)

sys.exit()









####################################################################################################
# some helper functions for plotting:
def get_figure(rows,cols):
    fig = plt.figure(figsize=(3.25*cols,3.0*rows))
    axes = fig.subplots(rows,cols)
    if rows==1 and cols==1:
        axes = [axes]
    return fig,*axes

def clip(v,low=0,high=1):
    return min(max(v,0),1)

def multiply_hex(h,factor):
    rgb = mpl.colors.to_rgb(h)
    rgb = [clip(c*factor) for c in rgb]
    hexc = mpl.colors.to_hex(rgb)
    return hexc

def bleaching_to_color(b):
    v = 0.5
    levels = []
    for k in range(8):
        levels.append(v)
        v = v*2
    levels = np.array(levels)
    cidx = np.argmin(np.abs(b-levels))%len(colors)
    return colors[cidx]
    
def bleaching_to_marker(b):
    v = 0.5
    levels = []
    for k in range(8):
        levels.append(v)
        v = v*2
    levels = np.array(levels)
    return markers[np.argmin(np.abs(b-levels))]


####################################################################################################
##### Get the data and set up the model objects

single_flash_data = ds.get_single_flash_data()

keys = list(single_flash_data.keys())
keys.sort()

r_1sec = om.Red('1sec')
b_1sec = om.Blue('1sec')
m_1sec = om.Magenta('1sec')

r_3sec = om.Red('3sec')
b_3sec = om.Blue('3sec')
m_3sec = om.Magenta('3sec')

####################################################################################################
##### Concept figure illustrating red, blue, and magenta models--1 second
# from fitting performed later:
# red     |(0, 8.0)|157.54|7.97|
# blue    |(0, 8.0)|201.39|5.85|0.33|
# magenta |(0, 8.0)|-39.77|213.49|8.80|0.13|
t,L = single_flash_data[(0,8.0)]

concept_fig,ax1,ax2 = get_figure(1,2)
#ax1,ax2 = concept_fig.subplots(1,2)

valid = np.where(np.logical_and(t>-0.2,t<0.95))
t = t[valid]
L = L[valid]
A_1_r = 157.74
tau_a_r = 7.97
A_1_b = 201.39
tau_a_b = 5.85
tau_b_b = 0.33
A_0_m = -39.77
A_1_m = 213.49
tau_a_m = 8.8
tau_b_m = 0.13

rfit = r_1sec.model(t,A_1_r,tau_a_r)
bfit = b_1sec.model(t,A_1_b,tau_a_b,tau_b_b)
mfit = m_1sec.model(t,A_0_m,A_1_m,tau_a_m,tau_b_m)

rres = L-rfit-50
bres = L-bfit-100
mres = L-mfit-150

ax1.plot(t,L,'k.',alpha=SCATTER_ALPHA,markersize=SCATTER_MARKERSIZE)
ax1.plot(t,rfit,'r-',alpha=FIT_ALPHA)
ax1.plot(t,bfit,'b-',alpha=FIT_ALPHA)
ax1.plot(t,mfit,color=MAIN_MODEL_COLOR,linestyle='-',alpha=FIT_ALPHA)

ax1.plot(t,rres,'r-')
ax1.plot(t,bres,'b-')
ax1.plot(t,mres,color=MAIN_MODEL_COLOR,linestyle='-')

ax1.set_yticks([-50,-100,-150])
ax1.set_yticklabels([])
ax1.grid(True,axis='y')
ax1.plot([-.2,-.2],[150,200],'k-',linewidth=3)
ax1.text(-.19,200,'$50\;\mathrm{nm}$',ha='left',va='top')
ax1.text(.2,100,'$%s$'%r_1sec.tex(),color='r')
ax1.text(.2,60,'$%s$'%b_1sec.tex(),color='b')
ax1.text(.2,20,'$%s$'%m_1sec.tex(),color=MAIN_MODEL_COLOR)

linestyles = [':','-','--']
#colors = ['r','g','b','c']
increments = {'A_0':10.0,'A_1':50.0,'tau_a':5.0,'tau_b':0.2}

for pidx,param in enumerate(m_1sec.parameter_names):
    for fidx,factor in enumerate([-1,0,1]):
        tex_param = param
        if param.find('tau')==0:
            tex_param = '\\'+param
        params = [t,A_0_m,A_1_m,tau_a_m,tau_b_m]
        params[pidx+1] = params[pidx+1]+factor*increments[param]
        dat = m_1sec.model(*params)-150*pidx
        if fidx==1:
            color = MAIN_MODEL_COLOR
            ax2.text(0,-150*pidx,'$%s$'%tex_param,ha='right',va='bottom',color='k')
        else:
            color = 'k'
        ax2.plot(t,dat,color+linestyles[fidx],label='$%s$'%param)
            
ax2.set_yticks([])
ax2.plot([-.2,-.2],[150,200],'k-',linewidth=3)
ax2.text(-.19,200,'$50\;\mathrm{nm}$',ha='left',va='top')
ax2.text(0.1,-450,'$%s$'%m_1sec.tex(),ha='left',va='bottom',color=MAIN_MODEL_COLOR)
ax1.set_xlabel('time (s)')
ax2.set_xlabel('time (s)')
ax1.set_xticks([0,.25,.5,.75])
ax2.set_xticks([0,.25,.5,.75])
plt.savefig('figs/concept_1sec.png',dpi=DEFAULT_PRINT_DPI)

####################################################################################################
##### Concept figure illustrating red, blue, and magenta models--3 seconds
# from fitting performed later:
# red     |(0, 8.0)|144.21|9.80|
# blue    |(0, 8.0)|168.98|7.22|0.09|
# magenta |(0, 8.0)|-40.06|206.61|9.14|0.07|

t,L = single_flash_data[(0,8.0)]

concept_fig,ax1,ax2 = get_figure(1,2)

valid = np.where(np.logical_and(t>-0.2,t<=2.9))
t = t[valid]
L = L[valid]
A_1_r = 144.21
tau_a_r = 9.80
A_1_b = 168.98
tau_a_b = 7.22
tau_b_b = 0.09
A_0_m = -40.06
A_1_m = 206.61
tau_a_m = 9.14
tau_b_m = 0.07

rfit = r_3sec.model(t,A_1_r,tau_a_r)
bfit = b_3sec.model(t,A_1_b,tau_a_b,tau_b_b)
mfit = m_3sec.model(t,A_0_m,A_1_m,tau_a_m,tau_b_m)

rres = L-rfit-50
bres = L-bfit-100
mres = L-mfit-150

ax1.plot(t,L,'k.',alpha=SCATTER_ALPHA,markersize=SCATTER_MARKERSIZE)
ax1.plot(t,rfit,'r-',alpha=FIT_ALPHA)
ax1.plot(t,bfit,'b-',alpha=FIT_ALPHA)
ax1.plot(t,mfit,color=MAIN_MODEL_COLOR,linestyle='-',alpha=FIT_ALPHA)

ax1.plot(t,rres,'r-')
ax1.plot(t,bres,'b-')
ax1.plot(t,mres,color=MAIN_MODEL_COLOR,linestyle='-')

ax1.set_yticks([-50,-100,-150])
ax1.set_yticklabels([])
ax1.grid(True,axis='y')
ax1.plot([-.2,-.2],[150,200],'k-',linewidth=3)
ax1.text(-.17,200,'$50\;\mathrm{nm}$',ha='left',va='top')
ax1.text(.2,100,'$%s$'%r_1sec.tex(),color='r')
ax1.text(.2,60,'$%s$'%b_1sec.tex(),color='b')
ax1.text(.2,20,'$%s$'%m_1sec.tex(),color=MAIN_MODEL_COLOR)

linestyles = [':','-','--']
#colors = ['r','g','b','c']
increments = {'A_0':20.0,'A_1':50.0,'tau_a':6.0,'tau_b':0.1}

for pidx,param in enumerate(m_1sec.parameter_names):
    for fidx,factor in enumerate([-1,0,1]):
        tex_param = param
        if param.find('tau')==0:
            tex_param = '\\'+param
        params = [t,A_0_m,A_1_m,tau_a_m,tau_b_m]
        params[pidx+1] = params[pidx+1]+factor*increments[param]
        dat = m_1sec.model(*params)-150*pidx
        if fidx==1:
            color = MAIN_MODEL_COLOR
            ax2.text(0,-150*pidx,'$%s$'%tex_param,ha='right',va='bottom',color='k')
        else:
            color = 'k'
        ax2.plot(t,dat,color+linestyles[fidx],label='$%s$'%param)
            
ax2.set_yticks([])
ax2.plot([-.2,-.2],[150,200],'k-',linewidth=3)
ax2.text(-.17,200,'$50\;\mathrm{nm}$',ha='left',va='top')
ax2.text(0.1,-450,'$%s$'%m_1sec.tex(),ha='left',va='bottom',color=MAIN_MODEL_COLOR)
ax1.set_xlabel('time (s)')
ax2.set_xlabel('time (s)')
ax1.set_xticks([0,1,2])
ax2.set_xticks([0,1,2])
plt.savefig('figs/concept_3sec.png',dpi=DEFAULT_PRINT_DPI)



####################################################################################################
##### Fit the single-flash data with the magenta model and visualize results

# in all models, time constants should be positive
if True:
    r_1sec.set_bounds('tau_a',0.0,np.inf)
    b_1sec.set_bounds('tau_a',0.0,np.inf)
    b_1sec.set_bounds('tau_b',0.0,np.inf)
    r_3sec.set_bounds('tau_a',0.0,np.inf)
    b_3sec.set_bounds('tau_a',0.0,np.inf)
    b_3sec.set_bounds('tau_b',0.0,np.inf)
    m_1sec.set_bounds('tau_a',0.0,np.inf)
    m_1sec.set_bounds('tau_b',0.0,np.inf)
    m_3sec.set_bounds('tau_a',0.0,np.inf)
    m_3sec.set_bounds('tau_b',0.0,np.inf)

for k in keys:
    t = single_flash_data[k][0]
    L = single_flash_data[k][1]

    valid = np.where(np.logical_and(t>=-0.2,t<=0.95))
    tfit = t[valid]
    Lfit = L[valid]

    try:
        r_1sec.fit(tfit,Lfit,k,weighting_function=lambda x: np.ones(len(x)))
    except RuntimeError as re:
        print(re)
    try:
        b_1sec.fit(tfit,Lfit,k,weighting_function=lambda x: np.ones(len(x)))
    except RuntimeError as re:
        print(re)
    try:
        m_1sec.fit(tfit,Lfit,k,weighting_function=lambda x: np.ones(len(x)))
    except RuntimeError as re:
        print(re)

for k in keys:
    t = single_flash_data[k][0]
    L = single_flash_data[k][1]
    if t.max()<2.5:
        continue

    valid = np.where(np.logical_and(t>=-0.2,t<=3.0))
    tfit = t[valid]
    Lfit = L[valid]

    try:
        r_3sec.fit(tfit,Lfit,k,weighting_function=lambda x: np.ones(len(x)))
    except RuntimeError as re:
        print(re)
    try:
        b_3sec.fit(tfit,Lfit,k,weighting_function=lambda x: np.ones(len(x)))
    except RuntimeError as re:
        print(re)
    try:
        m_3sec.fit(tfit,Lfit,k,weighting_function=lambda x: np.ones(len(x)))
    except RuntimeError as re:
        print(re)



fig,row1,row2 = get_figure(2,2)
axes = [row1,row2]

keys = m_1sec.fitted_parameters.keys()

ymax = -np.inf
ymin = np.inf
tmax = -np.inf
tmin = np.inf
nozoom = 0
zoom = 1
for k in keys:
    subject = k[0]
    b = k[1]
    color = bleaching_to_color(b)
    fitcolor = multiply_hex(color,0.75)
    marker = bleaching_to_marker(b)
    t,y = m_1sec.data[k]
    ymin = min(np.min(y),ymin)
    ymax = max(np.max(y),ymax)
    params = m_1sec.fitted_parameters[k]
    fit = m_1sec.model(t,*[params[pn] for pn in m_1sec.parameter_names])
    
    axes[nozoom][subject].plot(t,y,marker=marker,markerfacecolor='none',markeredgecolor=color,alpha=0.5,linestyle='',markersize=SCATTER_MARKERSIZE)
    axes[nozoom][subject].plot(t,fit,color=fitcolor,marker='',linestyle='-',linewidth=1,alpha=1.0)
    axes[nozoom][subject].plot(t+1000,fit+100000,color=fitcolor,markeredgecolor=color,markerfacecolor='none',marker=marker,alpha=1,linestyle='-',markersize=4,label='%0.1f%%'%b)
    axes[zoom][subject].plot(t,y,marker=marker,markerfacecolor='none',markeredgecolor=color,alpha=0.5,linestyle='',markersize=SCATTER_MARKERSIZE)
    axes[zoom][subject].plot(t,fit,color=fitcolor,marker='',linestyle='-',linewidth=1,alpha=1.0)
    axes[zoom][subject].plot(t+1000,fit+100000,color=fitcolor,markeredgecolor=color,markerfacecolor='none',marker=marker,alpha=1,linestyle='-',markersize=4,label='%0.1f%%'%b)
    
    xlim = -0.2,0.95
    ylim = -50,ymax*1.1
    zxlim = -0.05,0.05
    zylim = -50,100


for idx,ax in enumerate(axes[0]):
    ax.legend(ncol=2,loc=2,fontsize=8)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if idx==0:
        ax.set_ylabel('$\Delta OPL$ (nm)')
    if idx>0:
        ax.set_yticklabels([])
    #ax.set_xlabel('time (s)')
    ax.grid(True)
for idx,ax in enumerate(axes[1]):
    ax.legend(ncol=2,loc=2,fontsize=8)
    ax.set_ylim(zylim)
    ax.set_xlim(zxlim)
    if idx==0:
        ax.set_ylabel('$\Delta OPL$ (nm)')
    if idx>0:
        ax.set_yticklabels([])
    ax.set_xlabel('time (s)')
    ax.grid(True)
    #ax.set_title('Subject %d (fit)'%(idx+1))
plt.savefig('./figs/fitting_single_flash_1s.png')


####################################################################################################
##### Plot fits for 3 sec measurements
fig,ax1 = get_figure(1,1)

keys = m_3sec.fitted_parameters.keys()

ymax = -np.inf
ymin = np.inf
tmax = -np.inf
tmin = np.inf
nozoom = 0
zoom = 1
for k in keys:
    subject = k[0]
    b = k[1]
    color = bleaching_to_color(b)
    fitcolor = multiply_hex(color,0.75)
    marker = bleaching_to_marker(b)
    t,y = m_3sec.data[k]
    ymin = min(np.min(y),ymin)
    ymax = max(np.max(y),ymax)
    params = m_3sec.fitted_parameters[k]
    fit = m_3sec.model(t,*[params[pn] for pn in m_1sec.parameter_names])
    
    ax1.plot(t,y,marker=marker,markerfacecolor='none',markeredgecolor=color,alpha=0.5,linestyle='',markersize=SCATTER_MARKERSIZE)
    ax1.plot(t,fit,color=fitcolor,marker='',linestyle='-',linewidth=1,alpha=1.0)
    ax1.plot(t+1000,fit+100000,color=fitcolor,markeredgecolor=color,markerfacecolor='none',marker=marker,alpha=1,linestyle='-',markersize=4,label='%0.1f%%'%b)
    
xlim = -0.2,2.9
ylim = -50,ymax*1.4

ax1.legend(ncol=2,loc=2,fontsize=8)
ax1.set_ylim(ylim)
ax1.set_xlim(xlim)
ax1.set_ylabel('$\Delta OPL$ (nm)')
ax1.set_xlabel('time (s)')
ax1.grid(True)
plt.savefig('./figs/fitting_single_flash_3s.png')




####################################################################################################
##### Plot RMS error as a function of bleach

markers1 = ['s','o']
markers3 = ['^','d']
fig,ax1 = get_figure(1,1)
used_markers = []

for k in m_1sec.rms_errors.keys():
    marker = 'g%s'%markers1[k[0]]
    if marker in used_markers:
        plt.semilogx(k[1],m_1sec.rms_errors[k],marker,markeredgecolor='k')
    else:
        plt.semilogx(k[1],m_1sec.rms_errors[k],marker,label='subject %d (1 s)'%(k[0]+1),markeredgecolor='k')
        used_markers.append(marker)

for k in m_3sec.rms_errors.keys():
    marker = 'g%s'%markers3[k[0]]
    if marker in used_markers:
        plt.semilogx(k[1],m_3sec.rms_errors[k],marker,markeredgecolor='k')
    else:
        plt.semilogx(k[1],m_3sec.rms_errors[k],marker,label='subject %d (3 s)'%(k[0]+1),markeredgecolor='k')
        used_markers.append(marker)


        
plt.legend()


plt.ylim((0,700))
plt.xlabel('bleaching %')
plt.ylabel('fitting error RMS (nm)')
plt.savefig('./figs/error_vs_bleach.png')


#####################################################################################################
##### Plot parameter behavior as a function of bleach

fig,[ax1,ax2],[ax3,ax4] = get_figure(2,2)

axes = [ax1,ax2,ax3,ax4]
params = ['A_0','A_1','tau_a','tau_b']

for ax,param in zip(axes,params):
    pfunc = ax.semilogx

    ax.set_title('$%s$'%param)
    for k in m_1sec.rms_errors.keys():
        marker = 'g%s'%markers1[k[0]]
        if marker in used_markers:
            pfunc(k[1],m_1sec.fitted_parameters[k][param],marker,markeredgecolor='k')
        else:
            pfunc(k[1],m_1sec.fitted_parameters[k][param],marker,label='subject %d (1 s)'%(k[0]+1),markeredgecolor='k')
            used_markers.append(marker)

    for k in m_3sec.rms_errors.keys():
        marker = 'g%s'%markers3[k[0]]
        if marker in used_markers:
            pfunc(k[1],m_3sec.fitted_parameters[k][param],marker,markeredgecolor='k')
        else:
            pfunc(k[1],m_3sec.fitted_parameters[k][param],marker,label='subject %d (3 s)'%(k[0]+1),markeredgecolor='k')
            used_markers.append(marker)

    if param in ['tau_a','tau_b']:
        ax.set_xlabel('bleach %')

plt.savefig('./figs/params_vs_bleach.png')
plt.show()
sys.exit()

#####################################################################################################
##### Plot residual error for high bleaching values over 1 sec and fit the residue with a second
##### exponential decay, effectively emulating Pandiyan 2022 (PNAS)
fig,ax1,ax2 = get_figure(1,2)
axes = [ax1,ax2]

keys = m_1sec.fitted_parameters.keys()

ymax = -np.inf
ymin = np.inf
tmax = -np.inf
tmin = np.inf
nozoom = 0
zoom = 1
error_m_1sec = om.Red('err 1 sec')
error_m_1sec.set_bounds('tau_a',0.0,np.inf)

for k in keys:
    subject = k[0]
    b = k[1]
    if b<32:
        continue
    t,_ = m_1sec.data[k]
    y = m_1sec.errors[k]
    valid = np.where(t<0.95)
    t = t[valid]
    y = y[valid]
    
    error_m_1sec.fit(t,y,k)
    
error_m_1sec.plot_fits()

for k in keys:
    color = bleaching_to_color(b)
    fitcolor = multiply_hex(color,0.75)
    marker = bleaching_to_marker(b)
    t,y = m_1sec.error[k]
    ymin = min(np.min(y),ymin)
    ymax = max(np.max(y),ymax)

    

    
    params = m_1sec.fitted_parameters[k]
    fit = m_1sec.model(t,*[params[pn] for pn in m_1sec.parameter_names])
    
    axes[nozoom][subject].plot(t,y,marker=marker,markerfacecolor='none',markeredgecolor=color,alpha=0.5,linestyle='',markersize=SCATTER_MARKERSIZE)
    axes[nozoom][subject].plot(t,fit,color=fitcolor,marker='',linestyle='-',linewidth=1,alpha=1.0)
    axes[nozoom][subject].plot(t+1000,fit+100000,color=fitcolor,markeredgecolor=color,markerfacecolor='none',marker=marker,alpha=1,linestyle='-',markersize=4,label='%0.1f%%'%b)
    axes[zoom][subject].plot(t,y,marker=marker,markerfacecolor='none',markeredgecolor=color,alpha=0.5,linestyle='',markersize=SCATTER_MARKERSIZE)
    axes[zoom][subject].plot(t,fit,color=fitcolor,marker='',linestyle='-',linewidth=1,alpha=1.0)
    axes[zoom][subject].plot(t+1000,fit+100000,color=fitcolor,markeredgecolor=color,markerfacecolor='none',marker=marker,alpha=1,linestyle='-',markersize=4,label='%0.1f%%'%b)
    
    xlim = -0.2,0.95
    ylim = -50,ymax*1.1
    zxlim = -0.05,0.05
    zylim = -50,100


for idx,ax in enumerate(axes[0]):
    ax.legend(ncol=2,loc=2,fontsize=8)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if idx==0:
        ax.set_ylabel('$\Delta OPL$ (nm)')
    if idx>0:
        ax.set_yticklabels([])
    #ax.set_xlabel('time (s)')
    ax.grid(True)
for idx,ax in enumerate(axes[1]):
    ax.legend(ncol=2,loc=2,fontsize=8)
    ax.set_ylim(zylim)
    ax.set_xlim(zxlim)
    if idx==0:
        ax.set_ylabel('$\Delta OPL$ (nm)')
    if idx>0:
        ax.set_yticklabels([])
    ax.set_xlabel('time (s)')
    ax.grid(True)
    #ax.set_title('Subject %d (fit)'%(idx+1))
plt.savefig('./figs/fitting_single_flash_1s.png')



plt.show()

