#!/usr/bin/env python3

import sys
# Load directories containing source codes
sys.path.append('../src/')

import read_parameters 
import time
import pytorch_eigen_solver

# Train neural networks

Param = read_parameters.Param()

# Set random seed, different processors start from different seeds
seed = 3905 # + int(time.time()) 

eig_solver = pytorch_eigen_solver.eigen_solver(Param, seed)
eig_solver.run()

