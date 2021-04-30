#!/usr/bin/env python3

import sys
# Load directories containing source codes
sys.path.append('../src/')

import read_parameters 
import training_data
import MyComm 

# Train neural networks

Param = read_parameters.Param()
Comm = MyComm.Comm(Param.distributed_training)

# Set random seed, different processors start from different seeds
seed = 3905 + int(time.time()) + Comm.rank 

torch.set_printoptions(precision=20)

eig_solver = pytorch_eigen_solver.eigen_solver(Param, Comm, seed)
eig_solver.run()

