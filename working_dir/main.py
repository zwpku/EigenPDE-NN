#!/usr/bin/python3
import torch
import time

import sys
import os

# Load directories containing source codes
sys.path.append('../src/')
sys.path.append('../utils/')

import pytorch_eigen_solver 
import MyComm 
import read_parameters 
import data_processor 

# Create directory ./data, if not exist yet.
if not os.path.exists('./data'):
    os.makedirs('./data')
    print ("Directory ./data created.")

# Create directory ./fig, if not exist yet.
if not os.path.exists('./fig'):
    os.makedirs('./fig')
    print ("Directory ./fig created.")

# List of possible scripts 
script_name_list = ["../utils/eval_nn_on_grid.py", "../utils/eval_potential_on_grid.py", "../utils/FVD-1d.py", "../utils/FVD-2d.py -eps_monitor"]

# Run different tasks for different value of task_id :
#   0-3: Run one of the scripts in the above list
#   4:   Prepare training data 
#          (load MD data when namd_data_flag=True, or generate sample data when it is False)
#   5:   Solve eigenvalue PDE by training neural networks
task_id = 1

if task_id <= 3 :
    # Run one of the scripts above
    print ("Task: %d: Run the script: %s\n" % (task_id, script_name_list[task_id]) )
    os.system(script_name_list[task_id])
elif task_id == 4 :
    # Load MD data and save it to txt file
    Param = read_parameters.Param()
    Comm = MyComm.Comm(Param.distributed_training)
    data_proc = data_processor.PrepareData(Param) 
    data_proc.prepare_data()
elif task_id == 5 :
    # Train neural networks

    print ("Task %d: Training\n" % task_id)

    Param = read_parameters.Param()
    Comm = MyComm.Comm(Param.distributed_training)

    # Set random seed, different processors start from different seeds
    seed = 3905 + int(time.time()) + Comm.rank 

    torch.set_printoptions(precision=20)

    eig_solver = pytorch_eigen_solver.eigen_solver(Param, Comm, seed)
    eig_solver.run()

