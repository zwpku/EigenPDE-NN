#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math
from common import *

lc = ['b', 'r', 'k', 'c', 'm', 'y', 'k']
f_name_vec = ['cosine-dihedral', 'sinine-dihedral']
feature_filename = 'features_on_data.txt'
nbin = 100

# load feature data 
f_data_vec = np.loadtxt('../%s/data/%s' % (working_dir_name, feature_filename), skiprows=1)
f_dim = f_data_vec.shape[1] - 1
# Weights are stored in the last column
weights = f_data_vec[:,-1]
# Normalize weights 
tot_w = np.sum(weights)
weights /= tot_w

for idx in range(f_dim):
    #plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if idx < len(f_name_vec) :
        label_name = f_name_vec[idx]
    else :
        label_name = 'NoName' 

    plt.hist(f_data_vec[:,idx], bins=nbin, density=True, weights=weights, color=lc[0], label='%s' % label_name)

    ax.legend(bbox_to_anchor=(0.5, 0, 0.5, 0.5))
    #ax.set_ylim(0.01, 0.10)
#    ax.set_yticks(np.arange(-2.1, 2.6, step=0.3))
    fig.tight_layout()
    out_fig_name = '../%s/fig/feature_hist_along_traj_%s.eps' % (working_dir_name, idx)
    fig.savefig(out_fig_name)
    print ("Output plot for %dth column of feature file: %s" % (idx+1, out_fig_name))

