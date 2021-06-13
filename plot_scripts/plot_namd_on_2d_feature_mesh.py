#!/usr/bin/python3

from pylab import *
from numpy import *
import matplotlib.cm as cm
from common import *

if all_eig_flag :
    idx_vec = range(1, num_k+1)
    tot_num_k = num_k
else :
    idx_vec = [num_k]
    tot_num_k = 1

fig, ax = plt.subplots(1, tot_num_k, figsize=(9, 5.5))
fig.suptitle('Eigenfunctions, %s' % task_name)

tot_min = -0.3
tot_max = 0.3

sign_list = [1 for i in range(tot_num_k)]
sign_list[0] = 1
if tot_num_k > 1 :
    sign_list[1] = -1
if tot_num_k > 2 :
    sign_list[2] = -1

for i in range(len(idx_vec)) : 
  base_name = '../%s/data/%s' % (working_dir_name, eig_file_name_prefix)
  if all_eig_flag :
      base_name = '%s_feature_all' % base_name 

  data_file = open('%s_%d.txt' % (base_name, idx_vec[i]), 'r')

  xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
  ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]

  Z = np.loadtxt(data_file, skiprows=2)

  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(x,y)

  print (Z.shape, Z.min(), Z.max())

  if tot_num_k > 1 :
      nn_ax = ax[i]
  else :
      nn_ax = ax

  im = nn_ax.imshow( sign_list[i] * Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], origin='lower', interpolation='none' )

  nn_ax.set_title('NN, %dth' % (idx_vec[i]) )

  if i == 0:
    yticks(np.linspace(xmin, xmax, 5))
  else :
    plt.setp(nn_ax.get_yticklabels(), visible=False)

cax = fig.add_axes([0.92, 0.12, .04, 0.79])
#fig.colorbar(im, cax=cax, orientation='horizontal',cmap=cm.jet)
fig.colorbar(im, cax=cax, cmap=cm.jet)
#cax.tick_params(labelsize=10)

base_name = '../%s/fig/eigvec_nn_feature' % (working_dir_name)
if all_eig_flag :
    base_name = '%s_all' % base_name 

fig_name = '%s_%d.eps' % (base_name, num_k)

savefig(fig_name)

print ("Output figure: %s" % fig_name)

