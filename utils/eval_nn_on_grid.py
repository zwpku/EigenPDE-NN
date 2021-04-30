#!/usr/bin/python3
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

Param = read_parameters.Param()
PotClass = potentials.PotClass(Param.dim, Param.pot_id, Param.stiff_eps)

k = Param.k

# File name 
eig_file_name_prefix = Param.eig_file_name_prefix

dim = Param.dim

if dim == 1 :
    # Grid in R
    xmin = Param.xmin
    xmax = Param.xmax
    nx = Param.nx
    dx = (xmax - xmin) / nx

    cell_size = dx 
    ncell = nx 
    new_shape = (nx)
    x_axis = np.linspace(xmin, xmax, nx)
    str_header = '%f %f %d\n' % (xmin,xmax,nx)
    xvec = torch.tensor(x_axis, dtype=torch.float64).unsqueeze(1)

if dim == 2 :
    # Grid in R^2
    xmin = Param.xmin
    xmax = Param.xmax
    nx = Param.nx

    ymin = Param.ymin
    ymax = Param.ymax
    ny = Param.ny

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    cell_size = dx * dy
    ncell = nx * ny

    new_shape = (ny, nx)
    str_header = '%f %f %d\n%f %f %d\n' % (xmin,xmax, nx, ymin, ymax, ny)

    x_axis = np.linspace(xmin, xmax, nx)
    y_axis = np.linspace(ymin, ymax, ny)

    xvec = torch.tensor(np.transpose([np.tile(x_axis, len(y_axis)), np.repeat(y_axis, len(x_axis))]), dtype=torch.float64)

if dim > 2 :
    xmin = Param.xmin
    xmax = Param.xmax
    nx = Param.nx

    ymin = Param.ymin
    ymax = Param.ymax
    ny = Param.ny

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    cell_size = dx * dy
    ncell = nx * ny
    new_shape = (ny, nx)
    str_header = '%f %f %d\n%f %f %d\n' % (xmin,xmax,nx, ymin, ymax, ny)

    x1_axis = np.linspace(xmin, xmax, nx)
    x2_axis = np.linspace(ymin, ymax, ny)

    x1_vec = np.tile(x1_axis, len(x2_axis)).reshape(nx* ny, 1)
    x2_vec = np.repeat(x2_axis, len(x1_axis)).reshape(nx* ny, 1)

    # Randomly generate compoments in other dimensions
    x_other = np.random.normal(size=dim-2)
    #x_other = [0.0, 0.0, 0.0]
    #x_other = np.zeros(dim-2)
    print('Random value of x[2:%d]:\n' % dim, x_other)

    other_x = np.tile(x_other, nx* ny).reshape(nx* ny, dim-2)

    tmp = np.concatenate((x1_vec, x2_vec, other_x), axis=1)

    xvec = torch.tensor(tmp,  dtype=torch.float64)

# Load trained neural network
file_name = './data/%s.pt' % (eig_file_name_prefix)
model = torch.load(file_name)
model.eval()
print ("Neural network loaded\n")

# Evaluate neural network functions at states
Y_hat_all = model(xvec).detach().numpy()

if Param.namd_data_flag == False :
    # Weights 
    weights = exp(-0.5 * Param.beta * PotClass.V(xvec.numpy())).flatten()

if Param.all_k_eigs : # In this case, output the first k eigenfunctions
    for idx in range(k):
        Y_hat = Y_hat_all[:,idx] / (LA.norm(Y_hat_all[:,idx]) * math.sqrt(cell_size))

        eigen_file_name_output = './data/%s_all_%d.txt' % (eig_file_name_prefix, idx+1)
        np.savetxt(eigen_file_name_output, np.reshape(Y_hat, new_shape), header=str_header, comments="", fmt="%.10f")
        print("%dth eigen function is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

        if Param.namd_data_flag == False :
            # Save the conjugated function
            Y_hat = Y_hat * weights 
            # print ('%.4e, %.4e' % (min(weights), max(weights)))
            Y_hat = Y_hat / (LA.norm(Y_hat) * math.sqrt(cell_size))

            eigen_file_name_output = './data/%s_all_%d_conjugated.txt' % (eig_file_name_prefix, idx+1)
            np.savetxt(eigen_file_name_output, np.reshape(Y_hat, new_shape), header=str_header, comments="", fmt="%.10f")
            print("\t%s" % (eigen_file_name_output))

else : # Output the kth eigenfunction

    # Read the c vector
    cvec_file_name = './data/cvec.txt' 
    cvec=np.loadtxt(cvec_file_name, skiprows=1)
    print ('Vector c: ', cvec)

    # Linear combination, and normalization
    Y_hat = Y_hat_all.dot(cvec)
    Y_hat = Y_hat / (LA.norm(Y_hat) * math.sqrt(cell_size))

    # Save as a 1d or 2d function 
    eigen_file_name_output = './data/%s_%d.txt' % (eig_file_name_prefix, k)
    np.savetxt(eigen_file_name_output, np.reshape(Y_hat, new_shape), header=str_header, comments="", fmt="%.10f")
    print("%dth eigen function is stored to:\n\t%s" % (k, eigen_file_name_output))

    if Param.namd_data_flag == False :
        # Save the conjugated function
        Y_hat = Y_hat * weights
        Y_hat = Y_hat / (LA.norm(Y_hat) * math.sqrt(cell_size))
        eigen_file_name_output = './data/%s_%d_conjugated.txt' % (eig_file_name_prefix, k)

        np.savetxt(eigen_file_name_output, np.reshape(Y_hat, new_shape), header=str_header, comments="", fmt="%.10f")
        print("\t%s" % (eigen_file_name_output))

