#!/usr/bin/python3
import torch
import time

import sys
import os

# Create directory ./data, if not exist yet.
if not os.path.exists('./data'):
    os.makedirs('./data')
    print ("Directory ./data created.")

# Create directory ./fig, if not exist yet.
if not os.path.exists('./fig'):
    os.makedirs('./fig')
    print ("Directory ./fig created.")

# List of possible scripts 
script_name_list = ["../utils/prepare_data.py", "../utils/train_nn.py", "../utils/eval_nn_on_grid.py", "../utils/eval_potential_on_grid.py", "../utils/FVD-1d.py", "../utils/FVD-2d.py -eps_monitor"]

script_info_list = ["Prepare training data",  
        "Solve eigenvalue PDE by training neural networks", 
        "Evaluate neural network on 1d or 2d grid", 
        "Evaluate potential on grid", 
        "Solve 1D eigenvalue PDE by finite volume method", 
        "Solve 2D eigenvalue PDE by finite volume method"]

task_id = 1

if task_id == 1 :
    import configparser
    # Read parameters from config file
    config = configparser.ConfigParser()
    config.read_file(open('params.cfg'))
    # If num_processor > 1, the training will be run in parallel. 
    # In this case, pytorch with mpi is needed. 
    num_processor = config['Training'].getint('num_processor')
    if num_processor > 1 :
        script_name_list[1] = "mpirun -np %d %s" % (num_processor, script_name_list[1])

# Run one of the scripts above
print ("Task %d: %s, \nCommand: %s\n" % (task_id, script_info_list[task_id], script_name_list[task_id]) )

os.system(script_name_list[task_id])

