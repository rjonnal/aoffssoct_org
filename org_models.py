import numpy as np
from matplotlib import pyplot as plt
#import mpl_interactions.ipyplot as iplt
import scipy.io as sio
import scipy.optimize as spo
import scipy.signal as sps
import sys,os,glob
import octoblob.plotting_functions as opf
import matplotlib as mpl
from setup import *

#colors = ['#%s'%f for f in ["b86800","91bf40","2f6266","1c3048","3c1121"]]
#model_colors = dict(zip(['b','m','r','g'],["#2e294e","#b288c0","#e71d36","#034c3c"]))


model_colors = dict(zip(['b','m','r','g','c'],["#2e29ae","#d288d0","#e71d36","#13bc4c","#00CCCC"]))
model_colors = dict(zip(['b','m','r','g','c'],['b','m','r','g','c']))
model_linestyles = dict(zip(['b','m','r','g','c'],['-','-','-','-','-']))
figsize = (8,6)


figdir = './figs'
os.makedirs(figdir,exist_ok=True)

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
    
def bleaching_to_color0(b):
    r = np.log10(b*5)/np.log10(64.0*15)
    g = np.log10(b*5)/np.log10(64.0*15)
    b = 0.5#0.75-((r+g)/2.0)
    b = max(0,min(b,1))
    return [g,r,b]

def bleaching_to_marker(b):
    v = 0.5
    levels = []
    for k in range(8):
        levels.append(v)
        v = v*2
    levels = np.array(levels)
    return markers[np.argmin(np.abs(b-levels))]

def residual_error(t,L,model,result):
    fit = model(t,*result)
    err = L-fit
    rms = np.sqrt(np.sum(err**2))
    return err,rms
    
class Model:
    # parent class of all models
    # every model should do the following:
    # 1. provide a name/label
    # 2. provide a LaTeX representation
    # 3. have a function representing the model, ORG_model
    # 4. have a function to perform fitting, which takes
    #    x,y data and fits to the model described in 3. this
    #    function should also take a separate key, with
    #    which to store the results of the fits
    # 5. for a given key, store the data and the fitting parameters
    # 6. plot fits and/or parameter behavior


    def __init__(self,tag='',bounds_dict={}):
        self.fitted_parameters = {}
        self.errors = {}
        self.rms_errors = {}
        self.data = {}
        self.tag = tag
        for k in bounds_dict.keys():
            if k in self.parameter_names:
                lower,upper = bounds_dict[k]
                idx = self.parameter_names.index(k)
                self.bounds[0][idx] = lower
                self.bounds[1][idx] = upper

    def set_bounds(self,param,lower,upper):
        idx = self.parameter_names.index(param)
        self.bounds[0][idx] = lower
        self.bounds[1][idx] = upper
        
    def name(self):
        if len(self.tag):
            out = '%s_%s'%(self.__class__.__name__,self.tag)
        else:
            out = self.__class__.__name__
        out = out.lower()
        return out

    def fit(self,t,y,key):
        result = spo.curve_fit(self.model,t,y)
        param_dict = dict(zip(self.parameter_names,result[0]))
        self.fitted_parameters[key] = param_dict
        self.data[key] = (t,y)

    def plot_params(self,marker=lambda key: ['go','bs'][key[0]],x=lambda key:key[1],xlabel='bleaching',slabel='subject'):
        labeled = []
        figures = {}
        axes = {}
        
        for pn in self.parameter_names:
            figures[pn] = plt.figure()
            axes[pn] = figures[pn].subplots(1,1)

            for k in self.fitted_parameters.keys():
                if not (pn,k[0]) in labeled:
                    axes[pn].plot(x(k),self.fitted_parameters[k][pn],marker(k),label='%s %d'%(slabel,k[0]))
                    labeled.append((pn,k[0]))
                else:
                    axes[pn].plot(x(k),self.fitted_parameters[k][pn],marker(k))
            axes[pn].set_title(pn)
            axes[pn].set_ylabel(pn)
            axes[pn].set_xlabel(xlabel)
            axes[pn].legend()
            
            outfn = 'param_%s_%s.png'%(pn,self.name())
            plt.savefig(os.path.join(figdir,outfn),dpi=300)

    def print_parameters(self,key_heading='key',key_func=lambda x: x):
        n_params = len(self.parameter_names)
        fmt_head = '|%s|'+'%s|'*n_params
        heading = fmt_head%tuple([key_heading]+self.parameter_names)
        sep = '|---|'+'---|'*n_params
        fmt_row = '|%s|'+'%0.2f|'*n_params
        print('### %s'%self.name())
        print(heading)
        print(sep)
        for k in self.fitted_parameters.keys():
            print(fmt_row%tuple([str(key_func(k))]+[self.fitted_parameters[k][pn] for pn in self.parameter_names]))
        print()

    def plot_fits(self,xlim=None,ylim=None,zoom=False):
        keys = self.fitted_parameters.keys()
        n_subjects = 1+np.max(list(set([k[0] for k in keys])))
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(1,n_subjects)
        try:
            a = len(axes)
        except:
            axes = [axes]

        ymax = -np.inf
        ymin = np.inf
        tmax = -np.inf
        tmin = np.inf
        for k in keys:
            subject = k[0]
            b = k[1]
            color = bleaching_to_color(b)
            fitcolor = multiply_hex(color,0.75)
            #fitcolor = color#[a*0.5 for a in color]
            
            marker = bleaching_to_marker(b)
            t,y = self.data[k]
            ymin = min(np.min(y),ymin)
            ymax = max(np.max(y),ymax)
            tmin = min(np.min(t),tmin)
            tmax = max(np.max(t),tmax)
            
            params = self.fitted_parameters[k]
            fit = self.model(t,*[params[pn] for pn in self.parameter_names])
            axes[subject].plot(t,y,marker=marker,markerfacecolor='none',markeredgecolor=color,alpha=0.5,linestyle='',markersize=3)
            axes[subject].plot(t,fit,color=fitcolor,marker='',linestyle='-',linewidth=2,alpha=1.0)
            #axes[subject].plot(t+1000,fit+100000,color=fitcolor,markeredgecolor=color,markerfacecolor='none',marker=marker,alpha=1,linestyle='-',markersize=4,label='%0.1f%%'%b)
        if xlim is None:
            xlim = tmin-0.1,tmax+0.1
            
        if ylim is None:
            ylim = ymin-0.6*ymax,ymax*1.25


        if zoom:
            xlim = -0.05,0.05
            ylim = -100,100

            
        for idx,ax in enumerate(axes):
            ax.legend(ncol=2,loc=4)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            if idx==0:
                ax.set_ylabel('$\Delta OPL$ (nm)')
            if idx>0:
                ax.set_yticklabels([])
            ax.set_xlabel('time (s)')
            opf.despine(ax,'tbrl')
            ax.grid(True)
            ax.set_title('Subject %d (fit)'%(idx+1))
            if idx==0:
                thandle = ax.text(np.mean(xlim),ylim[1]*.97,'$%s$'%self.tex(),ha='center',va='top',fontsize=8)
                thandle.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor=self.color, linewidth=3))
        if zoom:
            outfn = 'fits_zoomed_%s.png'%self.name()
            print('![Model fits for %s (zoomed).](%s)'%(self.name(),os.path.join(figdir,outfn)))
        else:
            outfn = 'fits_%s.png'%self.name()
            print('![Model fits for %s.](%s)'%(self.name(),os.path.join(figdir,outfn)))
        plt.savefig(os.path.join(figdir,outfn),dpi=300)
        print()
        #plt.close()
        
    def plot_errors(self,xlim=None,ylim=None):
        keys = self.fitted_parameters.keys()
        n_subjects = len(list(set([k[0] for k in keys])))
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(1,n_subjects)
        try:
            a = len(axes)
        except:
            axes = [axes]

        ymax = -np.inf
        ymin = np.inf
        tmax = -np.inf
        tmin = np.inf
        for idx,k in enumerate(keys):
            subject = k[0]
            b = k[1]
            color = bleaching_to_color(b)
            fitcolor = multiply_hex(color,0.75)
            #fitcolor = color#[a*0.5 for a in color]
            
            marker = bleaching_to_marker(b)
            t,_ = self.data[k]
            offset = np.log(b)*50
            y = self.errors[k]+offset
            rms = self.rms_errors[k]
            ymin = min(np.min(y),ymin)
            ymax = max(np.max(y),ymax)
            tmin = min(np.min(t),tmin)
            tmax = max(np.max(t),tmax)
            
            axes[subject].plot(t,y,color=fitcolor,marker='',linestyle='-',linewidth=1,alpha=1.0,label='%0.1f%%'%b)
            axes[subject].text(t[-1],np.median(y[-20:-5]),r'$\sigma=%0.1f\;\mathrm{nm}$'%rms,ha='left',va='center',fontsize=8)


        ylim = (ymin-75,ymax+50)
        xlim = (tmin,tmax)
        for idx,ax in enumerate(axes):
            ax.legend(ncol=2,loc=4,fontsize=8)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            if idx==0:
                ax.set_ylabel('$\Delta OPL-\overline{\Delta OPL}}$ (nm)')
            ax.set_yticks([])
            ax.set_xlabel('time (s)')
            opf.despine(ax,'tbrl')
            ax.set_title('Subject %d (error)'%(idx+1))
            if idx==0:
                thandle = ax.text(np.mean(xlim)+.25,ylim[1]*.97,'$%s$'%self.tex(),ha='center',va='top',fontsize=8)
                thandle.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor=self.color, linewidth=3))
            ax.plot([tmin+100e-3,tmin+100e-3],[ylim[1]-50,ylim[1]],'k-',linewidth=5)
            #ax.axvline(tmin+100e-3,ymin=ymax+50,ymax=ymax+100,linewidth=10)
            #ax.axvline(tmin+100e-3,linewidth=10)
            ax.text(tmin+150e-3,ylim[1]-25,'50 nm',fontsize=8,ha='left',va='center')
        outfn = 'error_%s.png'%self.name()
        plt.savefig(os.path.join(figdir,outfn),dpi=300)
        print('![Fitting error for %s.](%s)'%(self.name(),os.path.join(figdir,outfn)))
        print()
        plt.close()
        
class Red(Model):

    def __init__(self,tag='',bounds_dict={}):
        self.parameter_names = ['A_1','tau_a']
        self.n_params = len(self.parameter_names)
        self.bounds = [[-np.inf]*self.n_params,[np.inf]*self.n_params]
        super().__init__(tag,bounds_dict)
        self.n_params = len(self.parameter_names)
        self.color = model_colors['r']
        self.linestyle = model_linestyles[self.color]        
        
    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,key,weighting_function=lambda x: np.ones(x.shape),p0=None):
        #self.bounds[0][1] = 2*L.min()
        #self.bounds[1][1] = 0.0
        result = spo.curve_fit(self.model,t,L,p0=p0,
                            bounds = self.bounds,sigma=weighting_function(t))

        A_1,tau_a = result[0]
        self.fitted_parameters[key] = dict(zip(self.parameter_names,[A_1,tau_a]))
        self.errors[key],self.rms_errors[key] = residual_error(t,L,self.model,result[0])
        self.data[key] = (t,L)
        return A_1,tau_a
    
    def tex(self):
        return r"""\Delta OPL(t) = A_1\left[1-e^{-\tau_a\cdot t}\right]"""
    
    def model(self,t,A_1,tau_a):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_1*(1.0-np.exp(-t*tau_a))*H
    
class Blue(Model):

    def __init__(self,tag='',bounds_dict={}):
        self.parameter_names = ['A_1','tau_a','tau_b']
        self.n_params = len(self.parameter_names)
        self.bounds = [[-np.inf]*self.n_params,[np.inf]*self.n_params]
        super().__init__(tag,bounds_dict)
        self.n_params = len(self.parameter_names)
        self.color = model_colors['b']
        self.linestyle = model_linestyles[self.color]        
        
    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,key,weighting_function=lambda x: np.ones(x.shape),p0=None):
        #self.bounds[0][1] = 2*L.min()
        #self.bounds[1][1] = 0.0
        result = spo.curve_fit(self.model,t,L,p0=p0,
                            bounds = self.bounds,sigma=weighting_function(t))

        A_1,tau_a,tau_b = result[0]
        self.fitted_parameters[key] = dict(zip(self.parameter_names,[A_1,tau_a,tau_b]))
        self.errors[key],self.rms_errors[key] = residual_error(t,L,self.model,result[0])
        self.data[key] = (t,L)
        return A_1,tau_a,tau_b
    
    def tex(self):
        return r"""\Delta OPL(t) = A_1\left[-e^{-\tau_a\cdot t}+e^{-\tau_b\cdot t}\right]"""
    
    def model(self,t,A_1,tau_a,tau_b):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_1*(-np.exp(-t*tau_a)+np.exp(-t*tau_b))*H


class Green(Model):

    def __init__(self,tag='',bounds_dict={}):
        self.parameter_names = ['A_1','tau_a','tau_b','t_0']
        self.n_params = len(self.parameter_names)
        self.bounds = [[-np.inf]*self.n_params,[np.inf]*self.n_params]
        super().__init__(tag,bounds_dict)
        self.n_params = len(self.parameter_names)
        self.color = model_colors['g']
        self.linestyle = model_linestyles[self.color]        
        
    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,key,weighting_function=lambda x: np.ones(x.shape),p0=None):
        #self.bounds[0][1] = 2*L.min()
        #self.bounds[1][1] = 0.0
        result = spo.curve_fit(self.model,t,L,p0=p0,
                            bounds = self.bounds,sigma=weighting_function(t))

        A_1,tau_a,tau_b,t_0 = result[0]
        self.fitted_parameters[key] = dict(zip(self.parameter_names,[A_1,tau_a,tau_b,t_0]))
        self.errors[key],self.rms_errors[key] = residual_error(t,L,self.model,result[0])
        self.data[key] = (t,L)
        return A_1,tau_a,tau_b,t_0
    
    def tex(self):
        return r"""\Delta OPL(t) = A_1\left[-e^{-\tau_a\cdot (t-t_0)}+e^{-\tau_b\cdot (t-t_0)}\right]"""
    
    def model(self,t,A_1,tau_a,tau_b,t_0):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_1*(-np.exp(-(t-t_0)*tau_a)+np.exp(-(t-t_0)*tau_b))*H

    
class Magenta(Model):

    def __init__(self,tag='',bounds_dict={}):
        self.parameter_names = ['A_0','A_1','tau_a','tau_b']
        self.n_params = len(self.parameter_names)
        self.bounds = [[-np.inf]*self.n_params,[np.inf]*self.n_params]
        super().__init__(tag,bounds_dict)
        self.n_params = len(self.parameter_names)
        self.color = model_colors['m']
        self.linestyle = model_linestyles[self.color]        
        
    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,key,weighting_function=lambda x: np.ones(x.shape),p0=None):
        
        def weighting_function0(t):
            w = np.ones(len(t))
            w[np.where(t<0.05)] = 0.1 # 0.1 works
            #w[np.where(t>0)] = t[np.where(t>0)]
            return w
        
        result = spo.curve_fit(self.model,t,L,p0=p0,
                            bounds = self.bounds,sigma=weighting_function(t))

        A_0,A_1,tau_a,tau_b = result[0]
        self.fitted_parameters[key] = dict(zip(self.parameter_names,[A_0,A_1,tau_a,tau_b]))
        self.errors[key],self.rms_errors[key] = residual_error(t,L,self.model,result[0])
        self.data[key] = (t,L)
        return A_0,A_1,tau_a,tau_b
    
    def tex(self):
        return r"""\Delta OPL(t) = A_0+A_1\left[-e^{-\tau_a\cdot t}+e^{-\tau_b\cdot t}\right]"""
    
    def model(self,t,A_0,A_1,tau_a,tau_b):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return (A_0+A_1*(-np.exp(-t*tau_a)+np.exp(-t*tau_b)))*H
    
class Cyan(Model):
    """Pandiyan et al., 2022"""
    def __init__(self,tag='',bounds_dict={}):
        self.parameter_names = ['A_0','A_1','tau_a','tau_b']
        self.n_params = len(self.parameter_names)
        self.bounds = [[-np.inf]*self.n_params,[np.inf]*self.n_params]
        super().__init__(tag,bounds_dict)
        self.n_params = len(self.parameter_names)
        self.color = model_colors['c']
        self.linestyle = model_linestyles[self.color]        
        
    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,key,weighting_function=lambda x: np.ones(x.shape),p0=None):
        #self.bounds[0][1] = 2*L.min()
        #self.bounds[1][1] = 0.0
        result = spo.curve_fit(self.model,t,L,p0=p0,
                            bounds = self.bounds,sigma=weighting_function(t))

        A_0,A_1,tau_a,tau_b = result[0]
        self.fitted_parameters[key] = dict(zip(self.parameter_names,[A_0,A_1,tau_a,tau_b]))
        self.errors[key],self.rms_errors[key] = residual_error(t,L,self.model,result[0])
        self.data[key] = (t,L)
        return A_0,A_1,tau_a,tau_b
    
    def tex(self):
        return r"""\Delta OPL(t) = A_0\left[1-e^{-\tau_a\cdot t}\right]+A_1\left[1-e^{-\tau_b\cdot t}\right]"""
    
    def model(self,t,A_0,A_1,tau_a,tau_b):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return (A_0*(1-np.exp(-t*tau_a))+A_1*(1-np.exp(-t*tau_b)))*H
    
class Magenta_original(Model):

    def __init__(self):
        self.parameter_names = ['A_1','A_2','tau_1','tau_2','tau_3']
        self.n_params = len(self.parameter_names)
        self.bounds = [[-np.inf]*self.n_params,[np.inf]*self.n_params]
        super().__init__()

    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,key):
        cidx = np.where((t>=-np.inf)*(t<=0.08))[0]
        eidx = np.where((t>=0.05)*(t<=np.inf))[0]

        # first, we fit the elongation portion using the full x and y inputs:
        result_elongation = spo.curve_fit(self.elongation,t[eidx],L[eidx],
                                          bounds = ([-np.inf,-np.inf,-np.inf],
                                                    [np.inf,np.inf,np.inf]))
        A_2,tau_b,tau_c = result_elongation[0]

        elongation_fit = self.elongation(t,A_2,tau_b,tau_c)
        error = L-elongation_fit
        result_contraction = spo.curve_fit(self.contraction,t[cidx],error[cidx],
                                           bounds = ([-np.inf,-np.inf],
                                                     [np.inf,np.inf]))
        A_1,tau_a = result_contraction[0]
        self.fitted_parameters[key] = dict(zip(self.parameter_names,[A_1,A_2,tau_a,tau_b,tau_c]))
        self.data[key] = (t,L)
        return A_1,A_2,tau_a,tau_b,tau_c
    
    def model(self,t,A_1,A_2,tau_a,tau_b,tau_c):
        return self.contraction(t,A_1,tau_a)+self.elongation(t,A_2,tau_b,tau_c)
    
    def tex(self):
        return r"""A_1\left[1-\frac{1}{e^{t\tau_a}}\right]+A_2\left[\frac{1}{e^{t\tau_b}}-\frac{1}{e^{t\tau_c}}\right]"""
    
    def contraction(self,t,A_1,tau_a):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_1*(1.0-1.0/np.exp(t*tau_a))*H

    def elongation(self,t,A_2,tau_b,tau_c):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_2*(1.0/np.exp(t*tau_b)-1.0/np.exp(t*tau_c))*H


class TwoEventModel:

    def __init__(self,t0=20e-3):
        self.t0 = t0
        
    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()
    
    def fit(self,t,L,plot=False,plot_label=''):
        result = spo.curve_fit(self.ORG_model,t,L)
        A_1,A_2,tau_a,tau_b,tau_c = result[0]
        if plot:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(t,L,'g.',alpha=0.25)
            plt.plot(t,self.ORG_model(t,A_1,A_2,tau_a,tau_b,tau_c),'k-')
            plt.xlim((-0.2,0.95))
            plt.subplot(1,2,2)
            plt.plot(t,L,'g.',alpha=0.25)
            plt.plot(t,self.contraction(t,A_1,tau_a))
            plt.plot(t,self.elongation(t-self.t0,A_2,tau_b,tau_c))
            if len(plot_label):
                plt.suptitle(plot_label.replace('_',' '))
                plt.savefig('%s.png'%plot_label)
            plt.close('all')
        return A_1,A_2,tau_a,tau_b,tau_c
    
    def fit_piecewise(self,t,L):
        cidx = np.where((t>=-np.inf)*(t<=0.08))[0]
        eidx = np.where((t>=0.05)*(t<=np.inf))[0]

        # first, we fit the elongation portion using the full x and y inputs:
        result_elongation = spo.curve_fit(self.elongation,t[eidx],L[eidx],
                                          bounds = ([-np.inf,-np.inf,-np.inf],
                                                    [np.inf,np.inf,np.inf]))
        A_2,tau_b,tau_c = result_elongation[0]

        elongation_fit = self.elongation(t,A_2,tau_b,tau_c)
        error = L-elongation_fit
        result_contraction = spo.curve_fit(self.contraction,t[cidx],error[cidx],
                                           bounds = ([-np.inf,-np.inf],
                                                     [np.inf,np.inf]))
        A_1,tau_a = result_contraction[0]
        return A_1,A_2,tau_a,tau_b,tau_c
    
    def ORG_model(self,t,A_1,A_2,tau_a,tau_b,tau_c):
        return self.contraction(t,A_1,tau_a)+self.elongation(t-self.t0,A_2,tau_b,tau_c)
    
    def name(self):
        return self.__class__.__name__
    
    def tex(self):
        return r"""A_1\frac{1}{e^{t\tau_a}}+A_2\left[\frac{1}{e^{(t-t_0)\tau_b}}-\frac{1}{e^{(t-t_0)\tau_c}}\right]"""
    
    def contraction(self,t,A_1,tau_a):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_1*(1.0/np.exp(t*tau_a))*H

    def elongation(self,t,A_2,tau_b,tau_c):
        H = np.zeros(t.shape)
        H[t>0.0]=1
        return A_2*(1.0/np.exp(t*tau_b)-1.0/np.exp(t*tau_c))*H

    






# ################################
# ## OLD STUFF, IGNORE ###########

    
# class Model_0(OneEventModel):

#     def tex(self):
#         return r"""A_1e^{t\tau_a}+A_2\left[e^{t\tau_b}-e^{t\tau_c}\right]"""
    
#     def contraction(self,t,A_1,tau_a):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return A_1*(np.exp(t*tau_a))*H

#     def elongation(self,t,A_2,tau_b,tau_c):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return A_2*(np.exp(t*tau_b)-np.exp(t*tau_c))*H

# class Model_1(OneEventModel):

#     def tex(self):
#         return r"""-A_1e^{t\tau_a}+A_2\left[e^{t\tau_b}-e^{t\tau_c}\right]"""
    
#     def contraction(self,t,A_1,tau_a):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return -A_1*(np.exp(t*tau_a))*H

#     def elongation(self,t,A_2,tau_b,tau_c):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return A_2*(np.exp(t*tau_b)-np.exp(t*tau_c))*H
    
# class Model_2(OneEventModel):

#     def tex(self):
#         return r"""-A_1\frac{1}{e^{t\tau_a}}+A_2\left[\frac{1}{e^{t\tau_b}}-\frac{1}{e^{t\tau_c}}\right]"""
    
#     def contraction(self,t,A_1,tau_a):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return -A_1*(1.0/np.exp(t*tau_a))*H

#     def elongation(self,t,A_2,tau_b,tau_c):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return A_2*(1.0/np.exp(t*tau_b)-1.0/np.exp(t*tau_c))*H


# class Model_3(OneEventModel):

#     def tex(self):
#         return r"""A_1\frac{1}{e^{t\tau_a}}+A_2\left[\frac{1}{e^{t\tau_b}}-\frac{1}{e^{t\tau_c}}\right]"""
    
#     def contraction(self,t,A_1,tau_a):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return A_1*(1.0/np.exp(t*tau_a))*H

#     def elongation(self,t,A_2,tau_b,tau_c):
#         H = np.zeros(t.shape)
#         H[t>0.0]=1
#         return A_2*(1.0/np.exp(t*tau_b)-1.0/np.exp(t*tau_c))*H
