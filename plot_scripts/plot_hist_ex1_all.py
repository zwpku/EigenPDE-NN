#!/usr/bin/env python3
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1,4, figsize=(21, 4.5), gridspec_kw={'wspace': 0.07})

dims = [2, 50, 100]
cols = [0, 1]
tot_min = 0.0
tot_max = 0.3
beta = 1.0

working_dir_name = 'working_dir_2d_MetastableRadius'
#working_dir_name = 'working_dir_2d'

data_file = open('../%s/data/pot.txt' % (working_dir_name), 'r')
xmin, xmax, nx = [ float (x) for x in data_file.readline().split() ]
ymin, ymax, ny = [ float (x) for x in data_file.readline().split() ]

Z = np.loadtxt(data_file)
dx = (xmax-xmin) / nx
dy = (ymax-ymin) / ny

density = exp(-1.0 * beta * Z)
density = density / (sum(density) * dx * dy)

im = ax[0].imshow(density, origin = "lower", extent=[xmin,xmax,ymin, ymax], cmap=cm.jet, vmin=tot_min , vmax=tot_max)
ax[0].set_title(r'$\frac{1}{Z}\exp(-\beta V_2)$', fontsize=30)

tot_min = 0
tot_max = 0.3

nx= 200
ny= 200
xmin = -3.0
xmax = 3.0
ymin = -3.0
ymax = 3.0
dx = (xmax-xmin) / nx
dy = (ymax-ymin) / ny

for idx in range(3) :
    states_file_name = '../big-data/states_%dd.txt' % (dims[idx])
    print ('reading histgram data from file: %s' % (states_file_name))

    X_vec = np.loadtxt(states_file_name, skiprows=1, usecols=cols)

    h = np.histogram2d(X_vec[:,0], X_vec[:,1], bins=[nx, ny], range=[[xmin,xmax],[ymin,ymax]])[0]
    s = sum(sum(h))
    im = ax[idx+1].imshow(h.T / (s * dx * dy), origin = "lower", extent=[xmin,xmax,ymin, ymax], cmap=cm.jet, vmin=tot_min , vmax=tot_max)
    #im = ax[idx+1].imshow(density, origin = "lower", extent=[xmin,xmax,ymin, ymax], cmap=cm.jet, vmin=tot_min , vmax=tot_max)

for idx in range(4):
    ax[idx].set_xlabel(r'$x_1$', fontsize=30, labelpad=-4)
    ax[idx].set_xticks([-2, 0, 2])
    ax[idx].set_yticks([-2, 0, 2])
    ax[idx].tick_params(axis='y', labelsize=28)
    ax[idx].tick_params(axis='x', labelsize=28)
    if idx > 0 :
        ax[idx].set_title(r'$d=%d$' % (dims[idx-1]), fontsize=30)

ax[0].set_ylabel(r'$x_2$', fontsize=30, labelpad=-20)

cax = fig.add_axes([0.90, 0.13, .017, 0.75])
fig.colorbar(im, cax=cax, cmap=cm.jet) #ticks=[0,0.1,0.2, 0.3])
cax.tick_params(labelsize=28)

fig_file_name = './ex1_hist_states_all.eps' 
fig.savefig(fig_file_name, bbox_inches='tight')
print("\nOutput of 2D histgram plot: %s\n" % fig_file_name)

