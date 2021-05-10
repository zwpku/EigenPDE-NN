#!/usr/bin/env python3

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math
from common import *

if dim != 1 : 
    print ("Error: can not plot 1d data (dim=%d)" % dim)
    sys.exit()

lc = ['b', 'r', 'k', 'c', 'm', 'y']

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

if all_eig_flag :
    idx_vec = range(1, num_k+1)
    tot_num_k = num_k
else :
    idx_vec = [num_k]
    tot_num_k = 1

if with_FVD_solution :
# load eigenfunctions computed from FVD
    for idx in range(len(idx_vec)) :
        if conjugated_eigvec_flag == 1 :
            data_file = open('../%s/data/%s_FVD_%d_conjugated.txt' % (working_dir_name, eig_file_name_prefix, idx_vec[idx]), 'r')
        else :
            data_file = open('../%s/data/%s_FVD_%d.txt' % (working_dir_name, eig_file_name_prefix, idx_vec[idx]), 'r')
        xmin, xmax, nx = [ float (x) for x in data_file.readline().split() ]
        xvec = np.linspace(xmin, xmax, nx)
        sol_by_FVD=np.loadtxt(data_file, skiprows=1)
        plt.plot(xvec, sol_by_FVD, color=lc[idx], label=r'$FVD, idx=%d$' % (idx_vec[idx]) )

sign_list = [1 for i in range(tot_num_k)]
sign_list[0] = 1
if len(sign_list) > 1 :
    sign_list[1] = -1

for idx in range(len(idx_vec)) :
    base_name = '../%s/data/%s' % (working_dir_name, eig_file_name_prefix)
    if all_eig_flag :
        base_name = '%s_all' % base_name 

    if conjugated_eigvec_flag == 1 :
        data_file = open('%s_%d_conjugated.txt' % (base_name, idx_vec[idx]), 'r')
    else :
        data_file = open('%s_%d.txt' % (base_name, idx_vec[idx]), 'r')

    xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
    xvec = np.linspace(xmin, xmax, nx)
    sol_by_nn=np.loadtxt(data_file)
    plt.plot(xvec, sign_list[idx] * sol_by_nn, color=lc[idx-1], linestyle=':', label=r'$NN, idx=%d$' % (idx_vec[idx]))

ax.legend(bbox_to_anchor=(0.5, 0, 0.5, 0.5))

#ax.set_ylim(-0.5, 0.5)
#ax.set_yticks(np.arange(-1.5, 3.0, step=0.5))
plt.title('Eigenfunctions, %s' % task_name, fontsize=20)
fig.tight_layout()

base_name = '../%s/fig/eigvec_nn_and_FVD' % (working_dir_name)
if all_eig_flag :
    base_name = '%s_all' % base_name 

if conjugated_eigvec_flag == 1 :
    fig_name = '%s_%d_conjugated.eps' % (base_name, num_k)
else :
    fig_name = '%s_%d.eps' % (base_name, num_k)

savefig(fig_name)

print ("output figure: %s" % fig_name)

