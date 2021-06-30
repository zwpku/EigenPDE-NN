#!/usr/bin/env python3
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from common import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

states_file_name = '../%s/data/%s.txt' % (working_dir_name, data_filename_prefix)
print ('reading trajectory data from file: %s' % (states_file_name))

X_vec=np.loadtxt(states_file_name,skiprows=1)
    h = np.histogram(X_vec, bins=nbin, range=[xmin, xmax])[0]

    xvec = np.linspace(xmin, xmax, nbin)
    dx = (xmax - xmin) / nbin
    plt.plot(xvec, h / (sum(h) * dx))
    plt.title('Prob. density of training data, %s' % task_name)
    fig_file_name = '../%s/fig/hist_states.eps' % (working_dir_name)
    fig.savefig(fig_file_name, bbox_inches='tight')
    print("\nhistgram plot of 1D sampled states is generated: %s\n" % fig_file_name)
elif dim >= 2:
    #when dim>2, load the first two components from the data file
    cols = [0, 1]
    fig = plt.figure(figsize=(9, 8))
    ax = plt.gca()
    X_vec = np.loadtxt(states_file_name, skiprows=1, usecols=cols)
    xmin = config['grid'].getfloat('xmin')
    xmax = config['grid'].getfloat('xmax')
    ymin = config['grid'].getfloat('ymin')
    ymax = config['grid'].getfloat('ymax')
#    ymin = -2.5
#    ymax = 2.5

    nx= 200
    ny= 200

    dx = (xmax-xmin) / nx
    dy = (ymax-ymin) / ny
    h = np.histogram2d(X_vec[:,0], X_vec[:,1], bins=[nx, ny], range=[[xmin,xmax],[ymin,ymax]])[0]
    s = sum(sum(h))
    tot_min = 0.0
    tot_max = 0.3
    im = plt.imshow(h.T / (s * dx * dy), origin = "lower", extent=[xmin,xmax,ymin, ymax], cmap=cm.jet, vmin=tot_min , vmax=tot_max)
    #ax.set_title('Prob. density of training data, %s' % task_name)
    plt.xlabel(r'$x_1$', fontsize=30)
    plt.ylabel(r'$x_2$', fontsize=30, labelpad=-15)
    plt.xticks([-3, -2, -1, 0, 1,2,3])

    #cax = fig.add_axes([0.08, 0.04, .84, 0.04])
    #fig.colorbar(im, cax=cax, orientation='horizontal',cmap=cm.jet)

    cax = fig.add_axes([0.88, 0.11, .04, 0.77])
    fig.colorbar(im, cax=cax, cmap=cm.jet)
    cax.tick_params(labelsize=28)
    ax.tick_params(axis='y', labelsize=27)
    ax.tick_params(axis='x', labelsize=27)

    fig_file_name = '../%s/fig/hist_states.eps' % (working_dir_name)
    fig.savefig(fig_file_name, bbox_inches='tight')
    print("\nOutput of 2D histgram plot: %s\n" % fig_file_name)

