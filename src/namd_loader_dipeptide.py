import matplotlib.pyplot as plt
import matplotlib as mpl
import MDAnalysis as mda 
from MDAnalysis import transformations
from MDAnalysis.lib.distances import calc_dihedrals
from MDAnalysis.analysis import align, rms 
from functools import reduce
import os

import math
import numpy as np


# This class loads MD data generated by NAMD, and then save it in txt file.
class namd_data_loader() :
    def __init__(self, Param, training_data=True) :

        # Initialize parameters  
        self.psf_name = Param.psf_name

        self.which_data_to_use = Param.which_data_to_use
        self.use_biased_data = Param.use_biased_data
        self.weight_threshold_to_remove_states = Param.weight_threshold_to_remove_states
        self.load_data_how_often = Param.load_data_how_often
        self.align_data_flag =  Param.align_data_flag

        self.namd_data_filename_prefix = Param.namd_dcddata_filename_prefix

        self.training_data = training_data

        # Filename for output 
        if training_data == True :
            self.data_filename_prefix = Param.data_filename_prefix
            self.namd_data_path = Param.namd_dcddata_path
        else :
            self.data_filename_prefix = Param.data_filename_prefix_validation
            self.namd_data_path = Param.namd_dcddata_path_validation

        print ('Paths: %s/%s' % (self.namd_data_path, self.namd_data_filename_prefix) )

        self.temp_T = Param.temp_T

        # Compute beta depending on temperature
        self.beta = Param.namd_beta

        self.load_dcd_file = True

    # Plot angle data to file
    def plot_angle_and_weight_on_grid(self):

        # Grid: [-pi, pi] x [-pi, pi] 
        numx = self.pmf_on_angle_mesh.shape[0]
        numy = self.pmf_on_angle_mesh.shape[1]
        dx = 360.0 / numx
        dy = 360.0 / numy

        angle_counter = np.zeros((numx, numy))
        weight_counter = np.zeros((numx, numy))

        num_data = len(self.angles)

        fig, ax = plt.subplots(1,1)
        plt.plot(np.sort(self.weights))
        plt.yscale('log')
        filename = './fig/weights_sorted_in_1d.eps' 
        plt.savefig(filename)
        print ('\n[Info] 1d sorted weights saved to: %s' % filename)

        # Count histgram and weights 
        for i in range (num_data):

            idx = int ((self.angles[i, 0] + 180) / dx)
            idy = int ((self.angles[i, 1] + 180) / dy)

            angle_counter[idx, idy] += 1
            weight_counter[idx, idy] += self.weights[i]

        plt.clf()
        #plt.rc('text', usetex=True)
#        plt.rc('font', family='serif')

        fig, ax = plt.subplots(1,1)
        # Show the angle counting data in 2d plot
        h = ax.imshow(angle_counter.T, extent=[-180,180, -180, 180], cmap='jet', origin='lower')
        plt.colorbar(h)
        filename = './fig/count_of_angles.eps' 
        plt.savefig(filename)
        print ('\n[Info] Plot of 2d count data saved to: %s' % filename)

        plt.clf()
        fig, ax = plt.subplots(1,1)
        # Plot may not be smooth at states where counter is zero
        h = ax.imshow(weight_counter.T, extent=[-180,180, -180, 180], cmap='jet', origin='lower', norm=mpl.colors.LogNorm())
        plt.colorbar(h)
        filename = './fig/tot_weights_of_angles.eps' 
        plt.savefig(filename)

        print ('[Info] Plot of total weight for each 2d-cell saved to: %s\n' % filename)

        # Compute weights using PMF
        weights_on_angle_mesh = np.exp(-self.beta * self.pmf_on_angle_mesh)

        # Rescale weights by constant 
        rescale = np.sum(weights_on_angle_mesh * angle_counter) / num_data
        weights_on_angle_mesh /= rescale

        plt.clf()
        fig, ax = plt.subplots(1,1)

        # Plot may not be smooth at states where counter is zero
        #h = ax.imshow(weight_counter.T, extent=[-180,180, -180, 180], cmap='jet', origin='lower', norm=mpl.colors.LogNorm(), vmin=5e-8)
        h = ax.imshow(weights_on_angle_mesh.T, extent=[-180,180, -180, 180], cmap='jet', origin='lower', norm=mpl.colors.LogNorm(), vmin=5e-8)
        ax.set_xlabel(r'$\varphi$', fontsize=27, labelpad=-1, rotation=0)
        ax.set_ylabel(r'$\psi$', fontsize=27, labelpad=-5, rotation=0)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
        ax.set_yticks([-150, -100, -50, 0, 50, 100, 150])

        cbar = plt.colorbar(h, ax=ax, pad=0.03)
        cbar.set_ticks([1e-6, 1e-4, 1e-2, 1])
        cbar.ax.tick_params(labelsize=20)
        ax.set_title(r'weights', fontsize=27)

        filename = './fig/weights_of_angles.eps' 
        plt.savefig(filename, bbox_inches='tight')

        print ('[Info] Plot of 2d weight data saved to: %s\n' % filename)


    def load_angles_from_colvar_traj(self) :

        colvar_traj_filename = '%s/%so.colvars.traj' % (self.namd_data_path, self.namd_data_filename_prefix)
        fp = open(colvar_traj_filename)

        # Read values of colvars trajectory 
        angles = np.loadtxt(colvar_traj_filename, skiprows=2)

        # Only use angles of selected states
        self.angles = angles[::self.load_data_how_often, 1:]

        # For test purpose
        #print ('\nAngles of first 5 states:\n\t', self.angles[0:5,:])

    # Compute weights for each sampled data
    def weights_of_colvars_by_pmf(self) :
        # First, read PMF (Potential of Mean Force) on grid of angles
        colvar_pmf_filename = '%s/%so.pmf' % (self.namd_data_path, self.namd_data_filename_prefix)
        fp = open(colvar_pmf_filename)

        # The first line is not useful
        ch, num = fp.readline().split()

        # Second line
        ch, lbx, dx, numx, tmp = fp.readline().split()
        numx = int(numx)
        lbx, dx = float(lbx), float(dx) 

        # Third line
        ch, lby, dy, numy, tmp = fp.readline().split()
        numy = int(numy)
        lby, dy= float(lby), float(dy) 

        # Shift the boundary 
        lbx += dx * 0.5
        lby += dy * 0.5

        print ('[Info] 2d grid mesh for angles:\n\t[%.2f, %.2f], nx=%d\n\t[%.2f, %.2f], ny=%d' % (lbx, lbx + numx * dx, numx, lby, lby + numx * dy, numy))

        # Read PMF data on mesh of angles 
        self.pmf_on_angle_mesh = np.loadtxt(colvar_pmf_filename)[:,2].reshape([numx, numy])

        K_angle = self.angles.shape[0] 

        # Compute 2d indices of angles 
        angle_idx = [[int ((self.angles[i][0] - lbx) / dx), int ((self.angles[i][1] - lby) / dy)] for i in range(K_angle)]

        # Make sure indices are within range
        for i in range(K_angle):
            angle_idx[i] = [min(numx-1, angle_idx[i][0]), min(numy-1, angle_idx[i][1])]

        # Obtain PMF for data according to indices on the grid 
        pmf_along_colvars = np.array([self.pmf_on_angle_mesh[angle_idx[i][0]][angle_idx[i][1]] for i in range(K_angle)])

        # Compute weights using PMF
        weights_along_colvars = np.exp(-self.beta * pmf_along_colvars)

        # Rescale weights by constant 
        rescale = np.sum(weights_along_colvars) / K_angle
        weights_along_colvars /= rescale

        print ('\n[Info] Length of pmf data along trajectory: %s\n\tmin-max of pmf=(%.3f, %.3f)' % (len(pmf_along_colvars), min(pmf_along_colvars), max(pmf_along_colvars)) )

        print ('\t(min,max,sum) of weights=(%.3e, %.3e, %.3e)\n' % (min(weights_along_colvars), max(weights_along_colvars), sum(weights_along_colvars) ) )

        self.weights = weights_along_colvars

    def total_weights_sub_regions(self) :

        num_of_regions = 4
        region_names = ['C7eq', 'C7ax', 'l-channel', 'r-channel']

        c7eq_index = reduce(np.logical_and, (self.angles[:,0] > -120, self.angles[:,0] < -70, self.angles[:,1] > 40, self.angles[:,1] < 130))
        c7ax_index = reduce(np.logical_and, (self.angles[:, 0] > 40, self.angles[:, 0] < 100, self.angles[:,1] < 0, self.angles[:,1] > -150))
        l_trans_index = reduce(np.logical_and, (self.angles[:,0] > -25, self.angles[:,0] < 25, self.angles[:,1] > -110, self.angles[:,1] < 0))
        r_trans_index = reduce(np.logical_and, (self.angles[:,0] > 115, self.angles[:,0] < 140, self.angles[:,1] > -170, self.angles[:,1] < -30))

        set_of_index_set = [c7eq_index, c7ax_index, l_trans_index, r_trans_index]

        print ('Name\tnum\t min-w\tmax-w\tsum-w\ttot-w\tpercent')

        for idx_set in  range(num_of_regions) :
            index_set = set_of_index_set[idx_set] 
            core_weight = np.sum(self.weights[index_set])
            if core_weight > 0 :
                print ('%8s: %6d\t%8.2e\t%.2e\t%.2e\t%.1f\t%.2e' %
                    (region_names[idx_set], len(self.weights[index_set]),
                        np.min(self.weights[index_set]),
                        np.max(self.weights[index_set]), core_weight,
                        self.weights.sum(), core_weight / self.weights.sum() ) )


    def load_namd_traj(self):
        # Names of files containing trajectory data
        psf_filename = '%s/%s.psf' % (self.namd_data_path, self.psf_name)
        pdb_filename = '%s/%s.pdb' % (self.namd_data_path, self.psf_name)
        traj_filename = '%s/%s.dcd' % (self.namd_data_path, self.namd_data_filename_prefix)
        u = mda.Universe(psf_filename, traj_filename)

        # Length of trajectory 
        K_total = len(u.trajectory)
        print ("[Info] Data file: %s\n\t%d states in datafile" % (traj_filename, K_total), flush=True)

        total_time_traj = len(u.trajectory) * u.coord.dt * 1e-3 

        print ( '\tLength of trajectory: %d, dt=%.2fps, total time = %.2fns' % (len(u.trajectory), u.coord.dt, total_time_traj) )
        print ("\tTime range of the trajectory: [%.2fps, %.2fps]\n" % (u.trajectory[0].time, u.trajectory[-1].time) )

        self.selected_atoms = None
        angle_col_index = None

        #Note: the indices are hard-coded, and depend on the psf file.
        #This will be changed in future.
        if self.which_data_to_use == 'all' : 
            # Select all atoms
            select_argument = "all"
            # Names of atoms related to dihedral angles are C, NT, CY, N, CA.
            # Indices of these 5 atoms among selected atoms
            angle_col_index = [0, 2, 12, 14, 16]
        elif self.which_data_to_use == 'nonh' : 
            # Select all atoms expect hydron (counting starts from 1)
            select_argument = "bynum 1 2 3 5 9 13 14 15 17 19"
            # Indices of the 5 atoms (see above) among selected atoms
            angle_col_index = [0, 2, 5, 7, 8]
        else : 
            # If which_data_to_use = 'angle_atoms' or 'angle', only include atoms related to two dihedral angles
            select_argument = "bynum 1 3 13 15 17"
            # Indices of the 5 atoms (see above) among selected atoms
            angle_col_index = [0, 1, 2, 3, 4]

        self.selected_atoms = u.select_atoms(select_argument)
        # Number of selected atoms
        self.atom_num = len(self.selected_atoms.names)

        K = len(u.trajectory[::self.load_data_how_often])

        print ( '\n[Info] How_often=%d...\n\tNames of %d selected atoms:\n\t %s\n\tNames of angle-related atoms:\n\t %s\n' % (self.load_data_how_often, self.atom_num, self.selected_atoms.names, self.selected_atoms.names[angle_col_index]) )

        print ( '[Info] Length of loaded trajectory: %d\n\tdt=%.2fps, total time = %.2fns' % (K, u.coord.dt * self.load_data_how_often, total_time_traj) )

        if self.load_dcd_file == False :
            return 

        if self.align_data_flag != 'none' : 
            # Align states by tranforming coordinates (e.g. rotation, translation...)
            #ref = mda.Universe(psf_filename, traj_filename)
            ref = mda.Universe(pdb_filename).select_atoms("bynum 1 3 13 15 17")

            if self.align_data_flag == 'trans' : 
                transform = mda.transformations.fit_translation(u.select_atoms("bynum 1 3 13 15 17"), ref) 
            else :
                transform = mda.transformations.fit_rot_trans(u.select_atoms("bynum 1 3 13 15 17"), ref) 

            u.trajectory.add_transformations(transform)

            print("[Info] Aligning data (%s)...done." % self.align_data_flag, flush=True)

        print ( '\n[Info] Loading trajectory data ...\n')
        # Change the 3d vector to 2d vector
        self.traj_data = np.array([self.selected_atoms.positions for ts in u.trajectory[::self.load_data_how_often]]).reshape((-1, self.atom_num * 3))


    def cut_states_with_small_weights(self) :

        eff_indices = self.weights >= self.weight_threshold_to_remove_states

        self.weights = self.weights[eff_indices]
        self.traj_data = self.traj_data[eff_indices]
        self.angles = self.angles[eff_indices]

        # Rescale weights (again) by constant 
        rescale = np.mean(self.weights)
        self.weights /= rescale

        print ("\n[Info] States whose weight is below %.2e are removed. %d states left." % (self.weight_threshold_to_remove_states, self.traj_data.shape[0]) )

        print ('\t(min,max,sum) of (renormalized) weights =(%.3e, %.3e, %.3e)\n' % (min(self.weights), max(self.weights), sum(self.weights) ) )

    def compute_weights(self) :
        if self.use_biased_data == True : 
            print("[Info] Load PMF along states\n", flush=True)
            # Compute weights along trajectory according to PMF value of angles
            self.weights_of_colvars_by_pmf() 
        else :
            K = self.angles.shape[0]
            print("[Info] Since data are unbiased, all weights are 1\n", flush=True)
            self.weights = np.ones(K) 

    def load_all(self):

        self.load_namd_traj()
        self.load_angles_from_colvar_traj() 
        self.compute_weights() 
        self.total_weights_sub_regions()

        if self.load_dcd_file == True :
            # Actual length of loaded data 
            K = self.traj_data.shape[0]
            print("[Info] In total, %d states have been loaded\n" % K, flush=True)

            if self.angles.shape[0] != K : 
                print ("colvars trajectoy (length=%d) does not match data (length=%d) " % (self.angles.shape[0], K))
                exit(1)

    def plot_namd_data(self):
        self.load_angles_from_colvar_traj() 
        self.compute_weights() 
        # Plot PMF on angle mesh
        self.plot_angle_and_weight_on_grid()

    # Load sampled MD data from file, and save it to txt file
    def save_namd_data_to_txt(self):

        if self.training_data == False :
            states_file_name = './data/%s.txt' % (self.data_filename_prefix)
            if os.path.isfile(states_file_name) :
                print ('[Info] found test data in: %s\n' % states_file_name)
            else :
                print ('Warning: data for validation does not exist: %s!\n' % states_file_name)

            angle_output_file = './data/angle_along_traj_validation.txt' 
            self.load_angles_from_colvar_traj() 
            print ( '[Info] Angles along trajectory are saved to file: %s\n' % angle_output_file)
            np.savetxt(angle_output_file, self.angles, header='%d' % (self.angles.shape[0]), comments="", fmt="%.10f")
            return 

        # filename of trajectory data 
        states_file_name = './data/%s.txt' % (self.data_filename_prefix)
        angle_output_file = './data/angle_along_traj.txt' 

        if os.path.isfile(states_file_name) :
            print ('Dcd file will not be loaded since data file already exists: %s!\n' % states_file_name)
            self.load_dcd_file = False 

        self.load_all() 

        K = self.angles.shape[0]

        print ( '[Info] Angles along trajectory are saved to file: %s\n' % angle_output_file)
        np.savetxt(angle_output_file, self.angles, header='%d' % (K), comments="", fmt="%.10f")

        if self.load_dcd_file == True :
            # Save trajectory data to txt file
            np.savetxt(states_file_name, np.concatenate((self.traj_data, self.weights.reshape((K,1))), axis=1), header='%d %d' % (K, 3 * self.atom_num), comments="", fmt="%.10f")

            print("[Info] Sampled data are stored to: %s" % states_file_name)

        # Save indices of atoms to txt file
        atom_ids_file = './data/atom_ids.txt' 
        np.savetxt(atom_ids_file, self.selected_atoms.ix_array, header='%d' % self.atom_num, comments="", fmt="%d")

        print("[Info] Atom indices are stored to: %s" % atom_ids_file)

