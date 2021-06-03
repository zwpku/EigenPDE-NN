#!/usr/bin/python3

import MDAnalysis as mda 
from MDAnalysis import transformations
from MDAnalysis.analysis import align, rms 

import matplotlib.pyplot as plt
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

def f_on_angle(xvec, angle, idx) :
    angle = angle / 180 * pi 
    #return angle[:,0] + idx * angle[:,1]
    return xvec[:, 0] + idx * xvec[:,1]

Param = read_parameters.Param()

namd_loader = namd_loader_dipeptide.namd_data_loader(Param, True) 
xvec, angles, weights = namd_loader.load_all()

print ("angle=", angles.shape)

xvec = torch.from_numpy(xvec).double()
K = xvec.shape[0]

print ("Length of Trajectory: %d\n" % K)

k = Param.k

# File name 
eig_file_name_prefix = Param.eig_file_name_prefix

# Load trained neural network
file_name = './data/%s.pt' % (eig_file_name_prefix)
model = torch.load(file_name)
model.eval()
print ("Neural network loaded\n")

b_tot_weights = sum(weights) 

num_test = 0
mean_list = np.zeros(num_test)
var_list = np.zeros(num_test)

for f_idx in range(num_test) :
    test_val = f_on_angle(xvec, angles, f_idx).numpy()
    mean_list[f_idx] = (test_val * weights).sum() / b_tot_weights 
    var_list[f_idx]  = (test_val**2 * weights).sum() / b_tot_weights - mean_list[f_idx] **2 

if num_test > 0 :
    print ("Means: ", mean_list) 
    print ("Vars: ", var_list) 

# Evaluate neural network functions at states
Y_hat_all = model(xvec).detach().numpy()

print (Y_hat_all.shape)

if Param.all_k_eigs : # In this case, output the first k eigenfunctions
    b_tot_weights = sum(weights) 
    mean_list = [(Y_hat_all[:,idx] * weights).sum() / b_tot_weights for idx in range(k)]
    var_list = [(Y_hat_all[:,idx]**2 * weights).sum() / b_tot_weights - mean_list[idx]**2 for idx in range(k)]

    print ("Means: ", mean_list) 
    print ("Vars: ", var_list) 

    for idx in range(k):
        eigen_file_name_output = './data/%s_on_data_all_%d.txt' % (eig_file_name_prefix, idx+1)
        np.savetxt(eigen_file_name_output, np.concatenate((angles, Y_hat_all[:, idx, None]), axis=1), header='%d' % K, comments="", fmt="%.10f")
        print("%dth eigen function along is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

else : # Output the kth eigenfunction
    # Read the c vector
    cvec_file_name = './data/cvec.txt' 
    cvec=np.loadtxt(cvec_file_name, skiprows=1)
    print ('Vector c: ', cvec)

    # Linear combination
    Y_hat = Y_hat_all.dot(cvec)
    Y_hat = Y_hat / LA.norm(Y_hat) * math.sqrt(K)

    eigen_file_name_output = './data/%s_on_data_%d.txt' % (eig_file_name_prefix, k)
    np.savetxt(eigen_file_name_output, np.concatenate((angles[:,1:], Y_hat), axis=1), header='%d' % K, comments="", fmt="%.10f")

    print("%dth eigen function is stored to:\n\t%s" % (k, eigen_file_name_output))

