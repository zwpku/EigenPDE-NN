#!/usr/bin/python3

from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from common import *

direction_vec = ['x', 'y', 'r']
xaxis_name = ['y', 'x', 'theta']
#0 : along x
#1 : along y
#2 : along r
along_x_y_r = 2
dir_str = direction_vec[along_x_y_r]

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
sign_list = [1 for i in range(num_k)]
sign_list[0] = -1

if all_eig_flag : # print the first
    eig_index_k = 2
else :  # print the kth
    eig_index_k = num_k

if with_FVD_solution :
    if conjugated_eigvec_flag == 1:
        data_file = open('../%s/data/%s_FVD_%d_conjugated.txt' % (working_dir_name, eig_file_name_prefix, eig_index_k), 'r')
    else :
        data_file = open('../%s/data/%s_FVD_%d.txt' % (working_dir_name, eig_file_name_prefix, eig_index_k), 'r')

    xmin_fvd, xmax_fvd, nx_fvd = [ float (x) for x in data_file.readline().split() ]
    ymin_fvd, ymax_fvd, ny_fvd = [ float (x) for x in data_file.readline().split() ]
    Z_fvd = np.loadtxt(data_file)

    dx_fvd = (xmax_fvd - xmin_fvd) / nx_fvd 
    dy_fvd = (ymax_fvd - ymin_fvd) / ny_fvd 

base_name = '../%s/data/%s' % (working_dir_name, eig_file_name_prefix)
if all_eig_flag :
  base_name = '%s_all' % base_name 

if conjugated_eigvec_flag == 1 :
  data_file = open('%s_%d_conjugated.txt' % (base_name, eig_index_k), 'r')
else :
  data_file = open('%s_%d.txt' % (base_name, eig_index_k), 'r')

xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]
Z = np.loadtxt(data_file)

dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny

lc = ['b', 'r', 'k', 'c', 'm', 'y', 'g']

val_vec = [0.5, 1.0, 1.3]
state_num = 100

for idx in range(len(val_vec)):
    if with_FVD_solution :
        z1d_fvd_vec = np.zeros(state_num)
        if along_x_y_r == 0 : # along x
            xvec = np.linspace(ymin_fvd, ymax_fvd, state_num)
            xx = np.stack((np.full(state_num, val_vec[idx]), xvec), -1)
        if along_x_y_r == 1 : # along y 
            xvec = np.linspace(xmin_fvd, xmax_fvd, state_num)
            xx = np.stack((xvec, np.full(state_num, val_vec[idx])), -1)
        if along_x_y_r == 2 : # along r 
            xvec = np.linspace(0, 2 * pi, state_num)
            xx = val_vec[idx] * np.stack((np.cos(xvec), np.sin(xvec)), -1)

        for idx_1d in range(len(xx)) :
            xidx = int((xx[idx_1d,0] - xmin_fvd - 1e-8) / dx_fvd)
            yidx = int((xx[idx_1d,1] - ymin_fvd - 1e-8) / dy_fvd)
            z1d_fvd_vec[idx_1d] = Z_fvd[yidx][xidx]

        plt.plot( xvec, z1d_fvd_vec, color=lc[idx], linestyle='-', label=r'$FVD, %s=%.1f$' % (dir_str, val_vec[idx]) )

    z1d_vec = np.zeros(state_num)
    if along_x_y_r == 0 : # along x
        xvec = np.linspace(ymin, ymax, state_num)
        xx = np.stack((np.full(state_num, val_vec[idx]), xvec), -1)
    if along_x_y_r == 1 : # along y 
        xvec = np.linspace(xmin, xmax, state_num)
        xx = np.stack((xvec, np.full(state_num, val_vec[idx])), -1)
    if along_x_y_r == 2 : # along r 
        xvec = np.linspace(0, 2 * pi, state_num)
        xx = val_vec[idx] * np.stack((np.cos(xvec), np.sin(xvec)), -1)

    for idx_1d in range(len(xx)) :
        xidx = int((xx[idx_1d, 0] - xmin - 1e-8) / dx)
        yidx = int((xx[idx_1d, 1] - ymin - 1e-8) / dy)
        z1d_vec[idx_1d] = Z[yidx][xidx]

    plt.plot( xvec, sign_list[0] * z1d_vec, color=lc[idx], linestyle=':', label=r'$NN, %s=%.1f$' % (dir_str, val_vec[idx]) )

ax.legend(bbox_to_anchor=(0.5, 0, 0.5, 0.5))

plt.title('Eigenfunctions for different %s, %s' % (dir_str, task_name))
plt.xlabel(xaxis_name[along_x_y_r])

base_name = '../%s/fig/eigvec_nn_and_FVD' % (working_dir_name)
if all_eig_flag :
    base_name = '%s_all' % base_name 

if conjugated_eigvec_flag == 1 : 
    fig_name = '%s_%d_along_%s_conjugated.eps' % (base_name, eig_index_k, dir_str) 
else :
    fig_name = '%s_%d_along_%s.eps' % (base_name, eig_index_k, dir_str)

savefig( fig_name )

print ("output figure: %s" % fig_name)

