#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math
import configparser

lc = ['b', 'r', 'k', 'c', 'm', 'y']
log_data_name_vec = ['tot_loss', 'loss_part_1', 'constraint-1', 'constraint-2']

working_dir_name = 'working_dir_100d_MetastableRadius'
#working_dir_name = 'working_dir_2d'
#working_dir_name = 'working_dir_100d_MetastableRadius_alpha50'
#working_dir_name = 'working_dir_100d_MetastableRadius_varyingAlpha'

# read parameters from config file
config = configparser.ConfigParser()
config.read_file(open('../%s/params.cfg' % working_dir_name))
log_filename = config['default'].get('log_filename')

# load log data 
log_info_vec = np.loadtxt('../%s/data/%s' % (working_dir_name, log_filename), skiprows=1)
ns = log_info_vec.shape[0] 
#xvec = np.linspace(0, ns * 10, ns)
xvec = log_info_vec[:,0]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5), gridspec_kw={'wspace': 0.35})

label_name = log_data_name_vec[0]

ax[0].plot(xvec, log_info_vec[:,1], color='k', label='%s' % label_name)
ax[0].set_title('Total loss', fontsize=24)
ax[0].set_xlabel(r'step',fontsize=26)
ax[0].set_xticks([0,1000, 3000, 5000])
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].set_ylim((1.0, 150))
ax[0].set_yscale('log')

ax[1].plot(xvec, log_info_vec[:,5], color='k', label='%s' % label_name)
ax[1].plot(xvec, log_info_vec[:,6], color='k', label='%s' % label_name)
ax[1].plot(xvec, log_info_vec[:,7], color='k', label='%s' % label_name)
ax[1].axhline(y=0.219,  ls=':', color='k') 
ax[1].axhline(y=0.764,  ls=':', color='k') 
ax[1].axhline(y=2.760,  ls=':', color='k') 
ax[1].set_title('Eigenvalues', fontsize=24)
ax[1].set_xlabel(r'step',fontsize=26)
ax[1].set_xticks([0,1000, 3000, 5000])
ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)
ax[1].set_ylim((0.1, 50))
ax[1].set_yscale('log')
ax[1].text(4000, 0.24, r'$\lambda_1$', horizontalalignment='right', verticalalignment='bottom', fontsize=20)
ax[1].text(4000, 0.79, r'$\lambda_2$', horizontalalignment='right', verticalalignment='bottom', fontsize=20)
ax[1].text(4000, 2.79, r'$\lambda_3$', horizontalalignment='right', verticalalignment='bottom', fontsize=20)

ax[2].plot(xvec, log_info_vec[:,4], color='k', label='%s' % label_name)
ax[2].set_title(r'Constraint', fontsize=24)
ax[2].set_xlabel(r'step',fontsize=26)
ax[2].tick_params(axis='x', labelsize=20)
ax[2].tick_params(axis='y', labelsize=20)
ax[2].set_xticks([0,1000, 3000, 5000])
ax[2].set_ylim((2e-4, 1))
ax[2].set_yscale('log')
ax[2].set_yticks([1e-3, 1e-2, 1e-1])

fig.tight_layout()
out_fig_name = './ex1_loss_constraints.eps' 
fig.savefig(out_fig_name, bbox_inches='tight')
print ("Output plot: %s" % (out_fig_name))


