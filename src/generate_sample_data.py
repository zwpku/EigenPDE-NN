#!/usr/bin/env python3

# Generate trajectory data by Euler-Maruyama scheme

import numpy as np
import math
import random

import potentials 
import read_parameters 

Param = read_parameters.Param()
PotClass = potentials.PotClass(Param.dim, Param.pot_id, Param.stiff_eps)

dim = Param.dim

# Step-size in the numerical scheme
delta_t = Param.delta_t 
# K: total numbers of states
K = Param.K
beta = Param.beta
# Filename of the trajectory data
data_filename_prefix = Param.data_filename_prefix

print ("beta=%.2f\tdelta_t=%.2e\n" % (beta, delta_t))

X0 = np.random.randn(dim)

# Number of steps in the initialization phase
burn_step = 10000
print ("First, burning, total step = %d" % burn_step)
for i in range(burn_step):
    xi = np.random.randn(dim)
    X0 = X0 - PotClass.grad_V(X0.reshape(1,dim)) * delta_t + np.sqrt(2 * delta_t / beta) * xi

# Sample data by simulating SDE
# The last column contains the importance sampling weights of the states
X_vec = np.zeros((K, dim+1))
print_step_interval = int(K / 10)

print ("Next, generate %d states" % K)
for i in range(K):
    xi = np.random.randn(dim)
    X0 = X0 - PotClass.grad_V(X0.reshape(1,dim)) * delta_t + np.sqrt(2 * delta_t / beta) * xi
    X_vec[i, 0:dim] = X0
    # Set weights to 1.0 in the last column
    X_vec[i][dim] = 1.0
    if i % print_step_interval == 0:
       print ("%4.1f%% finished." % (i / K * 100), flush=True)

states_file_name = './data/%s.txt' % (data_filename_prefix)
np.savetxt(states_file_name, X_vec, header='%d' % K, comments="", fmt="%.10f")
print("\nsampled data are stored to: %s" % states_file_name)

