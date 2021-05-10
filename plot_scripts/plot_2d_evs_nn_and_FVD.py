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

if with_FVD_solution == True :
    if tot_num_k > 1 :
        fig, ax = plt.subplots(2, tot_num_k, figsize=(9, 5.5))
    else :
        fig, ax = plt.subplots(1, 2, figsize=(9, 5.5))
    fig.suptitle('Eigenfunctions, %s' % task_name)
else :
    fig, ax = plt.subplots(1, tot_num_k, figsize=(9, 5.5))
    fig.suptitle('Eigenfunctions, %s' % task_name)

tot_min = -0.3
tot_max = 0.3

if with_FVD_solution :
    for i in range(len(idx_vec)) :
      if conjugated_eigvec_flag == 1 :
        data_file = open('../%s/data/%s_FVD_%d_conjugated.txt' % (working_dir_name, eig_file_name_prefix, idx_vec[i]), 'r')
      else :
        data_file = open('../%s/data/%s_FVD_%d.txt' % (working_dir_name, eig_file_name_prefix, idx_vec[i]), 'r')

      xmin, xmax, nx = [ float (x) for x in data_file.readline().split() ]
      ymin, ymax, ny = [ float (x) for x in data_file.readline().split() ]

      Z = np.loadtxt(data_file)

      x = np.linspace(xmin, xmax, nx)
      y = np.linspace(ymin, ymax, ny)

      if tot_num_k > 1 :
          fvd_ax = ax[0, i]
      else :
          fvd_ax = ax[i]

      im = fvd_ax.imshow( Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )
      fvd_ax.set_title('FVD, %dth' % (idx_vec[i]))

      if i == 0:
        yticks(np.linspace(xmin, xmax, 5))
      else :
        plt.setp(fvd_ax.get_yticklabels(), visible=False)

sign_list = [1 for i in range(tot_num_k)]
sign_list[0] = 1
if tot_num_k > 1 :
    sign_list[1] = -1
if tot_num_k > 2 :
    sign_list[2] = -1

for i in range(len(idx_vec)) : 
  base_name = '../%s/data/%s' % (working_dir_name, eig_file_name_prefix)
  if all_eig_flag :
      base_name = '%s_all' % base_name 

  if conjugated_eigvec_flag == 1 :
      data_file = open('%s_%d_conjugated.txt' % (base_name, idx_vec[i]), 'r')
  else :
      data_file = open('%s_%d.txt' % (base_name, idx_vec[i]), 'r')

  xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
  ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]

  Z = np.loadtxt(data_file, skiprows=0)

  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(x,y)

#  tot_min = Z.min() 
#  tot_max = Z.max() 
#  print (tot_min, tot_max)

  if with_FVD_solution :
      if tot_num_k > 1 :
          nn_ax = ax[1, i]
      else :
          nn_ax = ax[tot_num_k+i]
  else :
      if tot_num_k > 1 :
          nn_ax = ax[i]
      else :
          nn_ax = ax
  im = nn_ax.imshow( sign_list[i] * Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )

  nn_ax.set_title('NN, %dth' % (idx_vec[i]) )

  if i == 0:
    yticks(np.linspace(xmin, xmax, 5))
  else :
    plt.setp(nn_ax.get_yticklabels(), visible=False)

cax = fig.add_axes([0.92, 0.12, .04, 0.79])
#fig.colorbar(im, cax=cax, orientation='horizontal',cmap=cm.jet)
fig.colorbar(im, cax=cax, cmap=cm.jet)
#cax.tick_params(labelsize=10)

base_name = '../%s/fig/eigvec_nn_and_FVD' % (working_dir_name)
if all_eig_flag :
    base_name = '%s_all' % base_name 

if conjugated_eigvec_flag == 1 :
    fig_name = '%s_%d_conjugated.eps' % (base_name, num_k)
else :
    fig_name = '%s_%d.eps' % (base_name, num_k)

savefig(fig_name)

print ("output figure: %s" % fig_name)

