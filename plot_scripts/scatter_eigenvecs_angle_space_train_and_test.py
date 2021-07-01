#!/usr/bin/python3

from pylab import *
from numpy import *
import matplotlib.cm as cm
from common import *

def select_indices(angle_data, weights, min_w) :
    nbin = 100
    xbins = np.linspace(-180, 180, num=nbin)
    ybins = np.linspace(-180, 180, num=nbin)

    idx1 = np.searchsorted(xbins, angle_data[:,0], side='right')
    idx2 = np.searchsorted(ybins, angle_data[:,1], side='right')

    max_cout = 2
    counter_of_bins = np.zeros((nbin, nbin))
    ndata = len(angle_data[:,0])
    mask_of_data = weights > min_w #np.ones(ndata)

    for idx in range(ndata) :
        ii = idx1[idx]
        jj = idx2[idx]
        if counter_of_bins[ii][jj] < max_cout :
            counter_of_bins[ii][jj] += 1 
        else :
            mask_of_data[idx] = False

    return mask_of_data 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
#fig.suptitle('Eigenfunctions by NN, %s' % task_name)

angle_filename = '../%s/data/angle_along_traj.txt' % working_dir_name
angle_data_train = np.loadtxt(angle_filename, skiprows=1)

data_filename = '../%s/data/%s.txt' % (working_dir_name, data_filename_prefix)
weights = np.loadtxt(data_filename, skiprows=1, usecols=-1)
min_w = 1e-7

print ( '%d states, range of weights: [%.3e, %.3e]' % (len(weights), weights.min(), weights.max()) )

use_indices = select_indices(angle_data_train, weights, min_w)

print ('Training data size reduced to %d' % (len(angle_data_train[use_indices,0])) )

sig_vec = [1.0, 1.0]
vmins=[-8.3, -2.5]
vmaxs=[0.13, 1.1]

eig_idx = 1

data_filename = '../%s/data/%s_on_data_%d.txt' % (working_dir_name, eig_file_name_prefix, eig_idx+1)
data_file = open(data_filename, 'r')

eigenfun_data = np.loadtxt(data_file, skiprows=1) * sig_vec[eig_idx]
print ('eigenfun %d (train):, (min,max)=(%.4f, %.4f)' % (eig_idx+1, min(eigenfun_data), max(eigenfun_data)) ) 

nn_ax = ax[0]
sc = nn_ax.scatter(angle_data_train[use_indices,0], angle_data_train[use_indices,1], s=2.0, c=eigenfun_data[use_indices], vmin=vmins[eig_idx], vmax=vmaxs[eig_idx], cmap='jet')

nn_ax.set_title('%dth eigenfunction, train' % (eig_idx+1) , fontsize=27)

angle_filename = '../%s/data/angle_along_traj_validation.txt' % working_dir_name
angle_data_valid = np.loadtxt(angle_filename, skiprows=1)

data_filename = '../%s/data/%s.txt' % (working_dir_name, data_filename_prefix_validation)
weights = np.loadtxt(data_filename, skiprows=1, usecols=-1)
min_w = 1e-7

print ( '%d states, range of weights: [%.3e, %.3e]' % (len(weights), weights.min(), weights.max()) )

use_indices = select_indices(angle_data_valid, weights, min_w)

print ('Test data size reduced to %d' % (len(angle_data_valid[use_indices,0])) )

sig_vec = [1.0, 1.0]
data_filename = '../%s/data/%s_on_data_%d_validation.txt' % (working_dir_name, eig_file_name_prefix, eig_idx+1)
data_file = open(data_filename, 'r')

eigenfun_data = np.loadtxt(data_file, skiprows=1) * sig_vec[eig_idx]
print ('eigenfun %d (test):, (min,max)=(%.4f, %.4f)' % (eig_idx+1, min(eigenfun_data), max(eigenfun_data)) ) 

nn_ax = ax[1]
sc = nn_ax.scatter(angle_data_valid[use_indices,0], angle_data_valid[use_indices,1], s=2.0, c=eigenfun_data[use_indices],vmin=vmins[eig_idx], vmax=vmaxs[eig_idx], cmap='jet')

nn_ax.set_title('%dth eigenfunction, test' % (eig_idx+1) , fontsize=27)

cax = fig.add_axes([0.92, 0.10, .02, 0.80])
cbar = fig.colorbar(sc, cax=cax)
cbar.ax.tick_params(labelsize=20)
if eig_idx == 0 :
  cbar.set_ticks([-8.0, -6.0, -4.0, -2.0, 0])
else :
  cbar.set_ticks([-2.0, -1.0, 0, 1.0])

for i in range(2) : 
  ax[i].tick_params(axis='x', labelsize=18, pad=1.5)
  ax[i].tick_params(axis='y', labelsize=20, pad=0.0)
  ax[i].set_xlabel(r'$\varphi$', fontsize=25, labelpad=-1, rotation=0)
  ax[i].set_xticks([-150, -100, -50, 0, 50, 100, 150])
  ax[i].set_yticks([-150, -100, -50, 0, 50, 100, 150])
  ax[i].set_ylabel(r'$\psi$', fontsize=25, labelpad=-10, rotation=0)

base_name = '../%s/fig/scatter_nn_angle' % (working_dir_name)

fig_name = '%s_train_and_test_%d.pdf' % (base_name, eig_idx)

savefig(fig_name, dpi=200, bbox_inches='tight')

print ("output figure: %s" % fig_name)

