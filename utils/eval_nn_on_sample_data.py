#!/usr/bin/python3

import MDAnalysis as mda 
from MDAnalysis import transformations
from MDAnalysis.analysis import align, rms 

import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import random
import math
from numpy import exp
from math import pi
import torch

import sys
sys.path.append('../src/')

import potentials 
import read_parameters 

# load sampled MD data from file
def load_validation_data():

    # Names of files containing trajectory data
    psf_filename = '%s/%s.psf' % (Param.namd_validation_data_path, Param.psf_name)
    traj_filename = '%s/%s.dcd' % (Param.namd_validation_data_path, Param.namd_data_filename_prefix)
    u = mda.Universe(psf_filename, traj_filename)

    # Length of trajectory 
    K_total = len(u.trajectory)
    print ("[Info] Data file: %s\n\t%d states in datafile" % (traj_filename, K_total), flush=True)

    total_time_traj = len(u.trajectory) * u.coord.dt * 1e-3 

    print ( 'Length of trajectory: %d, dt=%.2fps, total time = %.2fns\n' % (len(u.trajectory), u.coord.dt, total_time_traj) )
    print ("Time range of the trajectory: [%.4f, %.4f]\n" % (u.trajectory[0].time, u.trajectory[-1].time) )

    if Param.align_data_flag == True : 
        # Align states by tranforming coordinates (e.g. rotation, translation...)
        ref = mda.Universe(psf_filename, traj_filename)
        ag = u.select_atoms('all')
        transform = mda.transformations.fit_rot_trans(ag, ref) 
        u.trajectory.add_transformations(transform)
        print("[Info] Aligning data...done.", flush=True)

    selected_atoms = None
    angle_col_index = None

    #Note: the indices are hard-coded, and depend on the psf file.
    #This will be changed in future.
    if Param.which_data_to_use == 'all' : 
        # Select all atoms
        selected_atoms = u.select_atoms("all")
        # Names of atoms related to dihedral angles are C, NT, CY, N, CA.
        # Indices of these 5 atoms among selected atoms
        angle_col_index = [0, 2, 12, 14, 16]
    elif Param.which_data_to_use == 'nonh' : 
        # Select all atoms expect hydron (counting starts from 1)
        selected_atoms = u.select_atoms("bynum 1 2 3 5 9 13 14 15 17 19")
        # Indices of the 5 atoms (see above) among selected atoms
        angle_col_index = [0, 2, 5, 7, 8]
    else : 
        # If which_data_to_use = 'angle_atoms' or 'angle', only include atoms related to two dihedral angles
        selected_atoms = u.select_atoms("bynum 1 3 13 15 17")
        # Indices of the 5 atoms (see above) among selected atoms
        angle_col_index = [0, 1, 2, 3, 4]

    # Number of selected atoms
    atom_num = len(selected_atoms.names)

    traj_data = np.array([selected_atoms.positions for ts in u.trajectory]).reshape((-1, atom_num * 3))
    # Actual length of data (should be the same as K_total above)
    K = traj_data.shape[0]
    print ("[Info] load data: %d states, dim=%d" % (K, 3 * atom_num), flush=True)

    colvar_pmf_filename = '%s/%so.colvars.traj' % (Param.namd_validation_data_path, Param.namd_data_filename_prefix)
    fp = open(colvar_pmf_filename)

    # Read values of colvars trajectory 
    angles = np.loadtxt(colvar_pmf_filename, skiprows=2)
    print ("[Info] angles of %d states loaded" % angles.shape[0])

    return K_total, torch.from_numpy(traj_data).double(), angles

Param = read_parameters.Param()

k = Param.k

# File name 
eig_file_name_prefix = Param.eig_file_name_prefix

K, xvec, angles = load_validation_data()

# Load trained neural network
file_name = './data/%s.pt' % (eig_file_name_prefix)
model = torch.load(file_name)
model.eval()
print ("Neural network loaded\n")

# Evaluate neural network functions at states
Y_hat_all = model(xvec).detach().numpy()

if Param.all_k_eigs : # In this case, output the first k eigenfunctions
    for idx in range(k):
        Y_hat = Y_hat_all[:, idx, None] / LA.norm(Y_hat_all[:, idx]) * math.sqrt(K)
        eigen_file_name_output = './data/%s_on_data_all_%d.txt' % (eig_file_name_prefix, idx+1)
        np.savetxt(eigen_file_name_output, np.concatenate((angles[:,1:], Y_hat), axis=1), header='%d' % K, comments="", fmt="%.10f")
        print("%dth eigen function along is stored to:\n\t%s" % (idx+1, eigen_file_name_output))

else : # Output the kth eigenfunction
    # Read the c vector
    cvec_file_name = './data/cvec.txt' 
    cvec=np.loadtxt(cvec_file_name, skiprows=1)
    print ('Vector c: ', cvec)

    # Linear combination
    Y_hat = Y_hat_all.dot(cvec)
    Y_hat = Y_hat / LA.norm(Y_hat) * math.sqrt(K)

    eigen_file_name_output = './data/%s_on_data_%d.txt' % (eig_file_name_prefix, k)
    np.savetxt(eigen_file_name_output, np.concatenate((angles[:,1:], Y_hat), axis=1), header='%d' % K, comments="", fmt="%.10f")

    print("%dth eigen function is stored to:\n\t%s" % (k, eigen_file_name_output))

