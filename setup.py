from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers = ['o','X','s','v','^','<','>','D']

def figsave(fig,fn):
    fig.savefig(fn,dpi=600)
    fig.savefig('/home/rjonnal/Dropbox/apps/Overleaf/ao_ff_ss_org/figures/%s'%fn,dpi=600)
