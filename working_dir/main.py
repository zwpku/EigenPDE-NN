#!/usr/bin/python3
import torch
import time

import sys
sys.path.append('../src/')

import pytorch_eigen_solver 
import MyComm 
import read_parameters 
import namd_loader_dipeptide

Param = read_parameters.Param()
Comm = MyComm.Comm(Param.distributed_training)

task_id = 1

if task_id == 0 :
    namd_loader = namd_loader_dipeptide.namd_data_loader(Param) 
    namd_loader.save_namd_data_to_txt()
else : 
    # Set random seed, different processors start from different seeds
    seed = 3905 + int(time.time()) + Comm.rank 

    torch.set_printoptions(precision=20)

    eig_solver = pytorch_eigen_solver.eigen_solver(Param, Comm, seed)
    eig_solver.run()

