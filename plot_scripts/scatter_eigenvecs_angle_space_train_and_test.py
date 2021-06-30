#!/usr/bin/python3

from pylab import *
from numpy import *
import matplotlib.cm as cm
from common import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

if all_eig_flag :
    idx_vec = range(1, num_k+1)
    tot_num_k = num_k
else :
    idx_vec = [num_k]
    tot_num_k = 1

fig, ax = plt.subplots(1, tot_num_k, figsize=(14, 5.5))
#fig.suptitle('Eigenfunctions by NN, %s' % task_name)

angle_filename = '../%s/data/angle_along_traj_validation.txt' % working_dir_name
angle_data = np.loadtxt(angle_filename, skiprows=1)

nbin = 100
xbins = np.linspace(-180, 180, num=nbin)
ybins = np.linspace(-180, 180, num=nbin)

idx1 = np.searchsorted(xbins, angle_data[:,0], side='right')
idx2 = np.searchsorted(ybins, angle_data[:,1], side='right')

max_cout = 3
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

use_indices = mask_of_data > 0

print ('Reduce data size from %d to %d' % (ndata, sum(mask_of_data)) )

for i in range(len(idx_vec)) : 
  base_name = '../%s/data/%s_on_data' % (working_dir_name, eig_file_name_prefix)
  if all_eig_flag :
      base_name = '%s_all' % base_name 
  data_file = open('%s_%d.txt' % (base_name, idx_vec[i]), 'r')

  eigenfun_data = np.loadtxt(data_file, skiprows=1) * -1.0
  print ('eigenfun %d:, (min,max)=(%.4f, %.4f)' % (i+1, min(eigenfun_data), max(eigenfun_data)) ) 

  if tot_num_k > 1 :
      nn_ax = ax[i]
  else :
      nn_ax = ax
  sc = nn_ax.scatter(angle_data[use_indices,0], angle_data[use_indices,1], s=1, c=eigenfun_data[use_indices])

  nn_ax.set_title('%dth eigenfunction' % (idx_vec[i]) , fontsize=20)

if num_k > 1 :
    for i in range(len(idx_vec)) : 
      #ax[i].set_xticks([-160, -100, 0, 100, 160])
      #ax[i].set_yticks([-160, -100, 0, 100, 160])
      ax[i].tick_params(axis='x', labelsize=18, pad=1.5)
      ax[i].tick_params(axis='y', labelsize=18, pad=0.0)
      ax[i].set_xlabel(r'$\varphi$', fontsize=20, labelpad=-1, rotation=0)
      ax[i].set_xticks([-150, -100, -50, 0, 50, 100, 150])
      ax[i].set_yticks([-150, -100, -50, 0, 50, 100, 150])

      ax[i].set_ylabel(r'$\psi$', fontsize=20, labelpad=-10, rotation=0)

cax = fig.add_axes([0.92, 0.12, .02, 0.79])
#fig.colorbar(cax=cax, orientation='horizontal',cmap=cm.jet)
cbar = fig.colorbar(sc, cax=cax)
cbar.ax.tick_params(labelsize=17)

#cax.tick_params(labelsize=10)

base_name = '../%s/fig/scatter_nn_angle' % (working_dir_name)
if all_eig_flag :
    base_name = '%s_all' % base_name 

fig_name = '%s_%d.pdf' % (base_name, num_k)

savefig(fig_name, dpi=200, bbox_inches='tight')

print ("output figure: %s" % fig_name)

