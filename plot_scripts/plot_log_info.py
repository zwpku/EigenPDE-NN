#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math
from common import *

lc = ['b', 'r', 'k', 'c', 'm', 'y', 'k']
log_data_name_vec = ['tot_loss', 'loss_part_1', 'constraint-1', 'constraint-2']

# load log data 
log_info_vec = np.loadtxt('../%s/data/%s' % (working_dir_name, log_filename), skiprows=1)

for idx in range(1, log_info_vec.shape[1]):
    #plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if idx < len(log_data_name_vec) :
        label_name = log_data_name_vec[idx-1]
    else :
        if all_eig_flag == True :
            label_name = '%dth_eigval' % (idx - 3)
        else :
            label_name = '%dth_eigval' % tot_num_k

    plt.plot(log_info_vec[:,idx], log_info_vec[:,idx], color=lc[idx-1], label='%s' % label_name)
    ax.legend(bbox_to_anchor=(0.5, 0, 0.5, 0.5))
    plt.yscale('log')
#    ax.set_ylim(-2.1, 2.70)
#    ax.set_yticks(np.arange(-2.1, 2.6, step=0.3))
    fig.tight_layout()
    out_fig_name = '../%s/fig/log_info_%d_%s.eps' % (working_dir_name, idx-1, label_name)
    fig.savefig(out_fig_name)
    print ("Output plot for %dth column of logfile: %s" % (idx-1, out_fig_name))

