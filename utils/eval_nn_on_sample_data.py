#!/usr/bin/python3

import MDAnalysis as mda 
from MDAnalysis import transformations
from MDAnalysis.analysis import align, rms 

from numpy import linalg as LA
import numpy as np
import random
import math
from numpy import exp
from math import pi
import torch

import sys
sys.path.append('../src/')

import potentials 
import read_parameters 
import namd_loader_dipeptide
import data_set

Param = read_parameters.Param()

states_filename = './data/%s.txt' % (Param.data_filename_prefix_validation)
dataset = data_set.MD_data_set.from_file(states_filename)

features = data_set.feature_tuple(Param.nn_features)
features.convert_atom_ix_by_file('./data/atom_ids.txt')

if features.num_features > 0 :
    dataset.set_features(features)

K = dataset.K

print ("Length of Trajectory: %d\n" % K)

k = Param.k

# File name 
eig_file_name_prefix = Param.eig_file_name_prefix

# Load trained neural network
file_name = './data/%s.pt' % (eig_file_name_prefix)
model = torch.load(file_name)
model.eval()
print ("Neural network loaded\n")

dataset.generate_minbatch(dataset.K, False)

# Evaluate neural network functions at states
Y_hat_all = model(dataset).detach().numpy()

if Param.all_k_eigs : # In this case, output the first k eigenfunctions
    weights = dataset.weights.numpy()
    b_tot_weights = sum(weights) 
    mean_list = [(Y_hat_all[:,idx] * weights).sum() / b_tot_weights for idx in range(k)]
    var_list = [(Y_hat_all[:,idx]**2 * weights).sum() / b_tot_weights - mean_list[idx]**2 for idx in range(k)]

    print ("Means: ", mean_list) 
    print ("Vars: ", var_list) 

    for idx in range(k):
        eigen_file_name_output = './data/%s_on_data_all_%d.txt' % (eig_file_name_prefix, idx+1)
        np.savetxt(eigen_file_name_output, Y_hat_all[:, idx], header='%d' % K, comments="", fmt="%.10f")
        print("%dth eigen function along trajectory is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

else : # Output the kth eigenfunction
    # Read the c vector
    cvec_file_name = './data/cvec.txt' 
    cvec=np.loadtxt(cvec_file_name, skiprows=1)
    print ('Vector c: ', cvec)

    # Linear combination
    Y_hat = Y_hat_all.dot(cvec)
    Y_hat = Y_hat / LA.norm(Y_hat) * math.sqrt(K)

    eigen_file_name_output = './data/%s_on_data_%d.txt' % (eig_file_name_prefix, k)
    np.savetxt(eigen_file_name_output, Y_hat, header='%d' % K, comments="", fmt="%.10f")

    print("%dth eigen function is stored to:\n\t%s" % (k, eigen_file_name_output))

