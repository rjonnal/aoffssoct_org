import numpy as np
from matplotlib import pyplot as plt

def transfer_function_components(freq,A0,A1,taua,taub):
    t1 = 1j*A0/(2*np.pi*freq)
    t2 = A1/(taua*(2*1j*np.pi*freq/taua+1))
    t3 = A1/(taub*(2*1j*np.pi*freq/taub+1))
    return t1,t2,t3


A0, A1, taua, taub = -42.3771,300.72,8.0146,0.0408

freq = np.arange(1e-2,30,1e-3)

t1,t2,t3 = transfer_function_components(freq,A0,A1,taua,taub)

tf1 = np.abs(t1+t2-t3)**2
tf2 = np.abs(t1)**2 + np.abs(t2)**2 + np.abs(t3)**2

tf1db = 10*np.log10(tf1)
tf2db = 10*np.log10(tf2)
t1db = 10*np.log10(np.abs(t1)**2)
t2db = 10*np.log10(np.abs(t2)**2)
t3db = 10*np.log10(np.abs(t3)**2)

plt.semilogx(freq,tf1db,label='correct')
plt.semilogx(freq,tf2db,label='approx')
plt.semilogx(freq,t1db,label='t1')
plt.semilogx(freq,t2db,label='t2')
plt.semilogx(freq,t3db,label='t3')
plt.legend()
plt.figure()
plt.semilogx(freq,tf2db-tf1db)
plt.show()
