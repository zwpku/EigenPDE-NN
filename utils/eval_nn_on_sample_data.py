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

import potentials 
import read_parameters 
import namd_loader_dipeptide
import data_set

Param = read_parameters.Param()

print ('use validation data [y/n]:') 
use_validation_data = input()

if use_validation_data == 'y': 
    states_filename = './data/%s.txt' % (Param.data_filename_prefix_validation)
else :
    states_filename = './data/%s.txt' % (Param.data_filename_prefix)

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
dataset.load_ref_state() 

# Evaluate neural network functions at states
Y_hat_all = model(dataset).detach().numpy()

weights = dataset.weights.numpy()
b_tot_weights = sum(weights) 
mean_list = [(Y_hat_all[:,idx] * weights).sum() / b_tot_weights for idx in range(k)]
var_list = [(Y_hat_all[:,idx]**2 * weights).sum() / b_tot_weights - mean_list[idx]**2 for idx in range(k)]

print ("Means: ", mean_list) 
print ("Vars: ", var_list) 

for idx in range(k):
    if use_validation_data == 'y': 
        eigen_file_name_output = './data/%s_on_data_%d_validation.txt' % (eig_file_name_prefix, idx+1)
    else :
        eigen_file_name_output = './data/%s_on_data_%d.txt' % (eig_file_name_prefix, idx+1)

    np.savetxt(eigen_file_name_output, Y_hat_all[:, idx], header='%d' % K, comments="", fmt="%.10f")
    print("%dth eigen function along trajectory is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

