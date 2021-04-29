#!/usr/bin/env python3

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math
from common import *

fig = plt.figure(figsize=(8, 8))
ax = plt.gca()

data_file = open('../%s/data/pot.txt' % (working_dir_name), 'r')

if dim == 1 :
    xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
    xvec = np.linspace( xmin, xmax, nx, endpoint=True )
    pot_vec = np.loadtxt(data_file, skiprows=1)
    plt.plot( xvec, pot_vec, color='k' )
    plt.xlim(xmin, xmax)
    plt.ylim(-1.0, 5.0)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    plt.title('Potential, %s' % task_name, fontsize=20)
    fig_name = '../%s/fig/pot.eps' % (working_dir_name)
    fig.tight_layout()
    savefig(fig_name)
    print ("output figure: %s" % fig_name)

if dim >= 2 :
    xmin, xmax, nx = [ float (x) for x in data_file.readline().split() ]
    ymin, ymax, ny = [ float (x) for x in data_file.readline().split() ]

    Z = np.loadtxt(data_file, skiprows=2)

    tot_min = Z.min()
    tot_max = Z.max() 

#    tot_min = 0 
    tot_max = 10

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    im = ax.imshow(Z, cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min, vmax=tot_max, origin='lower', interpolation='none' )
    plt.title(r'$V_1(x)$', fontsize=20)
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)

    cax = fig.add_axes([0.90, 0.11, .04, 0.77])
    fig.colorbar(im, cax=cax, cmap=cm.jet)

    cax.tick_params(labelsize=15)

    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)

    fig_name = '../%s/fig/pot.eps' % (working_dir_name)
    savefig(fig_name)
    print ("output figure: %s" % fig_name)
