import torch
import time

import sys
import os

import subprocess 

HOME_DIR = f'{os.getcwd()}/../'

if os.environ.get('PYTHONPATH') :
    python_path = HOME_DIR+'/src/:'+HOME_DIR+'/utils:' + os.environ.get('PYTHONPATH')
else :
    python_path = HOME_DIR+'/src/:'+HOME_DIR+'/utils' 

env_map = {'PATH': os.environ.get('PATH'), 'PYTHONPATH':python_path}

# Create directory ./data, if not exist yet.
if not os.path.exists('./data'):
    os.makedirs('./data')
    print ("Directory ./data created.")

# Create directory ./fig, if not exist yet.
if not os.path.exists('./fig'):
    os.makedirs('./fig')
    print ("Directory ./fig created.")

# List of possible scripts 
script_name_list = ["/utils/prepare_data.py", "/utils/prepare_data.py", "/utils/plot_hist_weights_namd.py", "/utils/train_nn.py", "/utils/eval_nn_on_grid.py", "/utils/eval_nn_on_sample_data.py", "/utils/FVD-1d.py", "/utils/FVD-2d.py -eps_monitor"]

script_info_list = ["Prepare training data",  
        "Prepare test data",
        "Plot histgram and weights of namd data",
        "Solve eigenvalue PDE by training neural networks", 
        "Evaluate neural network on 1d or 2d grid", 
        "Evaluate neural network on sample data", 
        "Solve 1D eigenvalue PDE by finite volume method", 
        "Solve 2D eigenvalue PDE by finite volume method"]

print ('Task list:')
for idx in range(len(script_info_list)) :
    print ('%d: %s' % (idx, script_info_list[idx])) 

print ('\nChoose a task ([0-%d]): ' % (len(script_name_list)-1) )

task_id = int (input())

# Run one of the scripts above
print ("Task %d: %s, \nCommand: %s\n" % (task_id, script_info_list[task_id], script_name_list[task_id]), flush=True)

if task_id != 1 :
    subprocess.run(HOME_DIR + script_name_list[task_id], env=env_map)
else :
    subprocess.run([HOME_DIR + script_name_list[1], 'test'], env=env_map)


