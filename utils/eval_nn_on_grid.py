#!/usr/bin/env python3
from numpy import linalg as LA
import numpy as np
import random
import math
from numpy import exp
from math import pi
import torch

import potentials 
import read_parameters 
import data_set

Param = read_parameters.Param(use_sections={'grid', 'training'})
PotClass = potentials.PotClass(Param.dim, Param.pot_id, Param.stiff_eps)

assert Param.md_data_flag == False , 'This script is not for MD data'

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
    xvec = x_axis.reshape((ncell, 1))
    weights = np.ones(ncell) 
    dataset = data_set.data_set(xvec, weights)

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

    xvec = np.transpose([np.tile(x_axis, len(y_axis)), np.repeat(y_axis, len(x_axis))])

    weights = np.ones(ncell) 
    dataset = data_set.data_set(xvec, weights)

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

    xvec = np.concatenate((x1_vec, x2_vec, other_x), axis=1)

    weights = np.ones(ncell) 
    dataset = data_set.data_set(xvec, weights)

# Load trained neural network
file_name = './data/%s.pt' % (eig_file_name_prefix)
model = torch.load(file_name)
model.eval()
print ("Neural network loaded\n")

dataset.generate_minibatch(dataset.K, False)
# Evaluate neural network functions at states
Y_hat_all = model(dataset).detach().numpy()

weights = exp(- Param.beta * PotClass.V(xvec)).flatten()

for idx in range(k):
    tot_weight = weights.sum()
    mean_y = (Y_hat_all[:, idx] * weights).sum() / tot_weight
    var_y = ((Y_hat_all[:, idx] - mean_y)**2 * weights).sum() / tot_weight
    print ('(mean, var)=', mean_y, var_y)
    Y_hat = Y_hat_all[:,idx] / (LA.norm(Y_hat_all[:,idx]) * math.sqrt(cell_size))

    eigen_file_name_output = './data/%s_%d.txt' % (eig_file_name_prefix, idx+1)
    np.savetxt(eigen_file_name_output, np.reshape(Y_hat, new_shape), header=str_header, comments="", fmt="%.10f")
    print("%dth eigen function is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

    # Save the conjugated function
    Y_hat = Y_hat_all[:,idx] * weights 
    # print ('%.4e, %.4e' % (min(weights), max(weights)))
    Y_hat = Y_hat / (LA.norm(Y_hat) * math.sqrt(cell_size))

    eigen_file_name_output = './data/%s_%d_conjugated.txt' % (eig_file_name_prefix, idx+1)
    np.savetxt(eigen_file_name_output, np.reshape(Y_hat, new_shape), header=str_header, comments="", fmt="%.10f")
    print("\t%s" % (eigen_file_name_output))
