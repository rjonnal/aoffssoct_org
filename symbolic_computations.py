from sympy import symbols, exp, diff, Heaviside, solve, log, plot, latex, simplify, integrate, fourier_transform, Abs, lambdify
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def taufix(s):
    return s.replace('taua','\\tau_{a}').replace('taub','\\tau_{b}')


# Equation from paper:
#  \Delta OPL(t) = u(t)[A_{0}+A_{1}(-e^{-\tau_{a}\cdot t}+e^{-\tau_{b}\cdot t})]

A0,A1,taua,taub,t,f = symbols('A0 A1 taua taub t f')
taua,taub = symbols('taua taub',positive=True)
o = symbols('oo')
OPL = Heaviside(t)*(A0 + A1*(-exp(-taua*t)+exp(-taub*t)))

dOPL = diff(OPL,t)
print('dOPL=%s'%taufix(latex(dOPL)))
tmax = solve(dOPL,t,dict=True)[0][t]
print('t_{max}=%s'%taufix(latex(tmax)))
OPLmax = simplify(OPL.subs(t,tmax))
print('OPL_{max} = %s'%taufix(latex(OPLmax)))
IOPL = integrate(OPL,t)
print('\int OPL dt = %s'%taufix(latex(IOPL)))
TF = fourier_transform(OPL,t,f)
PS = Abs(TF)**2
print('|F[\Delta OPL]|^2(f) = %s'%taufix(latex(PS)))

params = {A0:-30e-9,A1:100e-9,taua:10,taub:1.2}

fixed_OPL = OPL.subs(params)
fixed_PS = PS.subs(params)
#plot(fixed_PS,f)

# numerical power spectrum
nOPL = lambdify(t,fixed_OPL)
nt = np.arange(-.05,2.0,2.5e-3)
nlength = nOPL(nt)
plt.plot(nt,nlength)
plt.show()
nPS = lambdify(f,fixed_PS)
nfreq = np.arange(0.5,30,0.5)
npower = nPS(nfreq)
plt.loglog(nfreq,npower)
plt.show()
