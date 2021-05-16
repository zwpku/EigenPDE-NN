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

fig, ax = plt.subplots(1, tot_num_k, figsize=(12, 5.5))
fig.suptitle('Eigenfunctions by NN, %s' % task_name)

print (tot_num_k, idx_vec)

for i in range(len(idx_vec)) : 
  base_name = '../%s/data/%s_on_data' % (working_dir_name, eig_file_name_prefix)
  if all_eig_flag :
      base_name = '%s_all' % base_name 
  data_file = open('%s_%d.txt' % (base_name, idx_vec[i]), 'r')

  eigenfun_data = np.loadtxt(data_file, skiprows=1)
  print ('min-max:' , min(eigenfun_data[:,2]), max(eigenfun_data[:,2])) 

  if tot_num_k > 1 :
      nn_ax = ax[i]
  else :
      nn_ax = ax
  sc = nn_ax.scatter(eigenfun_data[:,0], eigenfun_data[:,1], c=eigenfun_data[:,2] , vmin=-4.5, vmax=1.1)

  nn_ax.set_title('%dth eigenfunction' % (idx_vec[i]) )

if num_k > 1 :
    for i in range(len(idx_vec)) : 
      #ax[i].set_xticks([-160, -100, 0, 100, 160])
      #ax[i].set_yticks([-160, -100, 0, 100, 160])
      ax[i].tick_params(axis='x', labelsize=18, pad=1.5)
      ax[i].tick_params(axis='y', labelsize=18, pad=0.0)
      ax[i].set_xlabel(r'$\phi$', fontsize=20, labelpad=-1, rotation=0)
      if i == 0 : 
          ax[i].set_ylabel(r'$\psi$', fontsize=20, labelpad=-10, rotation=0)

cax = fig.add_axes([0.92, 0.12, .02, 0.79])
#fig.colorbar(cax=cax, orientation='horizontal',cmap=cm.jet)
fig.colorbar(sc, cax=cax)
#cax.tick_params(labelsize=10)

base_name = '../%s/fig/scatter_nn_angle' % (working_dir_name)
if all_eig_flag :
    base_name = '%s_all' % base_name 

fig_name = '%s_%d.eps' % (base_name, num_k)

savefig(fig_name)

print ("output figure: %s" % fig_name)

