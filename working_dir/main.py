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
import namd_loader_dipeptide

# List of possible scripts 
script_name_list = ["../utils/generate_sample_data.py", "../utils/eval_nn_on_grid.py", "../utils/eval_potential_on_grid.py", "../utils/FVD-1d.py", "../utils/FVD-2d.py -eps_monitor"]

# Run different tasks for different value of task_id :
#   0-4: Run one of the scripts in the above list
#   5:   Load MD data
#   6:   Solve eigenvalue PDE by training neural networks
task_id = 1

if task_id <= 4 :
    # Run one of the scripts above
    print ("Task: %d: Run the script: %s\n" % (task_id, script_name_list[task_id]) )
    os.system(script_name_list[task_id])
elif task_id == 5 :
    # Load MD data and save it to txt file
    Param = read_parameters.Param()
    Comm = MyComm.Comm(Param.distributed_training)
    if Param.namd_data_flag == True : 
        print ("Task %d: Load MD data and save it to txt file\n" % task_id)
        namd_loader = namd_loader_dipeptide.namd_data_loader(Param) 
        namd_loader.save_namd_data_to_txt()
    else :
        print ("Error: To load MD data, the flag namd_data_flag should be True")
elif task_id == 6 :
    # Train neural networks

    print ("Task %d: Training\n" % task_id)

    Param = read_parameters.Param()
    Comm = MyComm.Comm(Param.distributed_training)

    # Set random seed, different processors start from different seeds
    seed = 3905 + int(time.time()) + Comm.rank 

    torch.set_printoptions(precision=20)

    eig_solver = pytorch_eigen_solver.eigen_solver(Param, Comm, seed)
    eig_solver.run()

