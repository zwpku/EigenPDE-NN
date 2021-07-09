#!/usr/bin/python3

import numpy as np
import math
from math import pi
import torch

import potentials 
import read_parameters 

Param = read_parameters.Param()

# File name 
eig_file_name_prefix = Param.eig_file_name_prefix
k = Param.k

# 2d angle Grid 
xmin = -math.pi
xmax = math.pi
nx = 100

ymin = -math.pi
ymax = math.pi
ny = 100

dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny

new_shape = (ny, nx)
str_header = '%f %f %d\n%f %f %d\n' % (xmin / math.pi * 180, xmax / math.pi * 180, nx, ymin / math.pi * 180, ymax / math.pi * 180, ny)

x_axis = np.linspace(xmin, xmax, nx)
y_axis = np.linspace(ymin, ymax, ny)

angle_vec = np.transpose([np.tile(x_axis, len(y_axis)), np.repeat(y_axis, len(x_axis))])
xvec = torch.from_numpy(np.column_stack((np.cos(angle_vec[:,0]), np.sin(angle_vec[:,0]), np.cos(angle_vec[:,1]), np.sin(angle_vec[:,1]))))

# Load trained neural network
file_name = './data/%s.pt' % (eig_file_name_prefix)
model = torch.load(file_name)
model.eval()
print ("Neural network loaded\n")

# Evaluate neural network functions at states
Y_hat_all = model.feature_forward(xvec).detach().numpy()

for idx in range(k):
    eigen_file_name_output = './data/%s_feature_all_%d.txt' % (eig_file_name_prefix, idx+1)
    np.savetxt(eigen_file_name_output, np.reshape(Y_hat_all[:,idx], new_shape), header=str_header, comments="", fmt="%.10f")
    print("%dth eigen function is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

