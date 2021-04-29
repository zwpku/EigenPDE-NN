#!/usr/bin/env python3

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math

# pot_id=5
# 2d, three wells along radius
def v_2d_3well_theta(theta):
# angle in [-pi, pi] 
  v_vec = np.zeros(len(theta))
  for idx in range(len(theta)) :
    # potential V_1
      if theta[idx] > pi / 3 : 
        v_vec[idx] = (1-(theta[idx] * 3 / pi- 1.0)**2)**2
      if theta[idx] < - pi / 3 : 
        v_vec[idx] = (1-(theta[idx] * 3 / pi + 1.0)**2)**2
      if theta[idx] > -pi / 3 and theta[idx] < pi / 3:
        v_vec[idx] = 3.0 / 5.0 - 2.0 / 5.0 * np.cos(3 * theta[idx])  
  return v_vec

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={'wspace': 0.28})

theta = np.linspace(-math.pi, math.pi, 200)
ax[0].plot(theta, v_2d_3well_theta(theta), linewidth=3)
ax[0].set_title(r'$V_0$', fontsize=28)
ax[0].set_xlabel(r'$\theta$', fontsize=25, labelpad=-1)
ax[0].tick_params(axis='y', labelsize=25)
ax[0].tick_params(axis='x', labelsize=25)
ax[0].set_xticks([-pi, -2*pi/3,0,2* pi/3, pi])
ax[0].set_xticklabels([r'$-\pi$', r'$-2\pi/3$',r'$0$',r'$2\pi/3$', r'$\pi$'])

working_dir_name = 'working_dir_2d_MetastableRadius'
data_file = open('../%s/data/pot.txt' % (working_dir_name), 'r')

xmin, xmax, nx = [ float (x) for x in data_file.readline().split() ]
ymin, ymax, ny = [ float (x) for x in data_file.readline().split() ]

Z = np.loadtxt(data_file, skiprows=2)

tot_min = Z.min()
tot_max = Z.max() 

#    tot_min = 0 
tot_max = 7
print (tot_min, tot_max)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)

im = ax[1].imshow(Z, cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min, vmax=tot_max, origin='lower', interpolation='none' )

theta = 2.0/3.0*math.pi
ax[1].annotate(r'$A$', (math.cos(theta), math.sin(theta)), color='w', size=30, va='center',ha='center')
theta = -2.0/3.0*math.pi
ax[1].annotate(r'$B$', (math.cos(theta), math.sin(theta)), color='w', size=30, va='center', ha='center')
theta = 0.0
ax[1].annotate(r'$C$', (math.cos(theta), math.sin(theta)), color='w', size=30, va='center', ha='left')

ax[1].set_title(r'$V_2$', fontsize=28)
ax[1].set_xlabel(r'$x_1$', fontsize=25, labelpad=-1)
ax[1].set_ylabel(r'$x_2$', fontsize=25, labelpad=-15)
ax[1].set_xticks([-3, -2, -1, 0, 1,2,3])
ax[1].tick_params(axis='y', labelsize=25)
ax[1].tick_params(axis='x', labelsize=25)

cax = fig.add_axes([0.90, 0.11, .035, 0.77])
cbar=fig.colorbar(im, cax=cax, cmap=cm.jet)

cax.tick_params(labelsize=28)
cbar.set_ticks([tot_min,2,4,6,8,10])
cbar.set_ticklabels([0,2,4,6,8,10])

fig_name = './ex1_pots_v0_v2.eps' 
savefig(fig_name)
print ("output figure: %s" % fig_name)
