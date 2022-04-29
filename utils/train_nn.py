#!/usr/bin/env python3

import read_parameters 
import time
import pytorch_eigen_solver

# Train neural networks

Param = read_parameters.Param(use_sections={'training'})

# Set random seed, different processors start from different seeds
seed = 3905 + int(time.time()) 

eig_solver = pytorch_eigen_solver.eigen_solver(Param, seed)
eig_solver.run()

