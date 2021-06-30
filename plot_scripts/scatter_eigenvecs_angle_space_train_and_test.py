#!/usr/bin/python3

from pylab import *
from numpy import *
import matplotlib.cm as cm
from common import *

def select_indices(angle_data) :
    nbin = 100
    xbins = np.linspace(-180, 180, num=nbin)
    ybins = np.linspace(-180, 180, num=nbin)

    idx1 = np.searchsorted(xbins, angle_data[:,0], side='right')
    idx2 = np.searchsorted(ybins, angle_data[:,1], side='right')

    max_cout = 2
    counter_of_bins = np.zeros((nbin, nbin))
    ndata = len(angle_data[:,0])
    mask_of_data = np.ones(ndata)

    for idx in range(ndata) :
        ii = idx1[idx]
        jj = idx2[idx]
        if counter_of_bins[ii][jj] < max_cout :
            counter_of_bins[ii][jj] += 1 
        else :
            mask_of_data[idx] = 0

    return mask_of_data > 0

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(2, 2, figsize=(13, 14))
#fig.suptitle('Eigenfunctions by NN, %s' % task_name)

angle_filename = '../%s/data/angle_along_traj.txt' % working_dir_name
angle_data_train = np.loadtxt(angle_filename, skiprows=1)

use_indices = select_indices(angle_data_train)

print ('Reduce (training) data size from %d to %d' % (len(angle_data_train[:,0]), len(angle_data_train[use_indices,0])) )

sig_vec = [1.0, 1.0]
vmins=[-8.3, -2.5]
vmaxs=[0.13, 1.1]

for i in range(2) : 
  data_filename = '../%s/data/%s_on_data_%d.txt' % (working_dir_name, eig_file_name_prefix, i+1)
  data_file = open(data_filename, 'r')

  eigenfun_data = np.loadtxt(data_file, skiprows=1) * sig_vec[i]
  print ('eigenfun %d (train):, (min,max)=(%.4f, %.4f)' % (i+1, min(eigenfun_data), max(eigenfun_data)) ) 

  nn_ax = ax[i,0]
  sc = nn_ax.scatter(angle_data_train[use_indices,0], angle_data_train[use_indices,1], s=2.0, c=eigenfun_data[use_indices], vmin=vmins[i], vmax=vmaxs[i], cmap='jet')

  nn_ax.set_title('%dth eigenfunction, train' % (i+1) , fontsize=27)

angle_filename = '../%s/data/angle_along_traj_validation.txt' % working_dir_name
angle_data_valid = np.loadtxt(angle_filename, skiprows=1)
use_indices = select_indices(angle_data_valid)

print ('Reduce (test) data size from %d to %d' % (len(angle_data_valid[:,0]), len(angle_data_valid[use_indices,0])) )

sig_vec = [1.0, 1.0]
for i in range(2) : 
  data_filename = '../%s/data/%s_on_data_%d_validation.txt' % (working_dir_name, eig_file_name_prefix, i+1)
  data_file = open(data_filename, 'r')

  eigenfun_data = np.loadtxt(data_file, skiprows=1) * sig_vec[i]
  print ('eigenfun %d (test):, (min,max)=(%.4f, %.4f)' % (i+1, min(eigenfun_data), max(eigenfun_data)) ) 

  nn_ax = ax[i,1]
  sc = nn_ax.scatter(angle_data_valid[use_indices,0], angle_data_valid[use_indices,1], s=2.0, c=eigenfun_data[use_indices],vmin=vmins[i], vmax=vmaxs[i], cmap='jet')

  nn_ax.set_title('%dth eigenfunction, test' % (i+1) , fontsize=27)

  if i == 0 :
      cax = fig.add_axes([0.92, 0.53, .02, 0.35])
      cbar = fig.colorbar(sc, cax=cax)
      cbar.ax.tick_params(labelsize=20)
      cbar.set_ticks([-8.0, -6.0, -4.0, -2.0, 0])
  else :
      cax = fig.add_axes([0.92, 0.11, .02, 0.35])
      cbar = fig.colorbar(sc, cax=cax)
      cbar.ax.tick_params(labelsize=20)
      cbar.set_ticks([-2.0, -1.0, 0, 1.0])

for i in range(2) : 
    for j in range(2) : 
      ax[i,j].tick_params(axis='x', labelsize=18, pad=1.5)
      ax[i,j].tick_params(axis='y', labelsize=20, pad=0.0)
      ax[i,j].set_xlabel(r'$\varphi$', fontsize=25, labelpad=-1, rotation=0)
      ax[i,j].set_xticks([-150, -100, -50, 0, 50, 100, 150])
      ax[i,j].set_yticks([-150, -100, -50, 0, 50, 100, 150])
      ax[i,j].set_ylabel(r'$\psi$', fontsize=25, labelpad=-10, rotation=0)

#cax = fig.add_axes([0.92, 0.12, .02, 0.79])
#cbar = fig.colorbar(sc, cax=cax)
#cbar.ax.tick_params(labelsize=17)

#cax.tick_params(labelsize=10)

base_name = '../%s/fig/scatter_nn_angle' % (working_dir_name)

fig_name = '%s_train_and_test.pdf' % (base_name)

savefig(fig_name, dpi=200, bbox_inches='tight')

print ("output figure: %s" % fig_name)

