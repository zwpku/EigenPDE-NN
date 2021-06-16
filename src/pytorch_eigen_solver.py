import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
import torch
from collections import OrderedDict
import time
from numpy import linalg as LA
import itertools 
from functools import reduce

import network_arch 

class eigen_solver():

    def __init__(self, Param, seed=3214):
        # The meanings of parameters are commented in the file: read_parameters.py 
        self.k = Param.k
        self.all_k_eigs = Param.all_k_eigs
        self.eig_w = Param.eig_w

        # When MD data is used, compute beta from temperature
        if Param.namd_data_flag == True : 
            self.beta = Param.namd_beta
        else :
            self.beta = Param.beta

        self.batch_uniform_weight = Param.batch_uniform_weight
        self.stage_list = Param.stage_list
        self.batch_size_list = Param.batch_size_list

        self.sort_eigvals_in_training = Param.sort_eigvals_in_training

        self.learning_rate_list = Param.learning_rate_list
        self.alpha_1_list = Param.alpha_1_list
        self.alpha_2_list = Param.alpha_2_list

        # Sizes of inner Layers (without input/output layers)
        self.arch_list = Param.inner_arch_size_list 

        self.use_Rayleigh_quotient = Param.use_Rayleigh_quotient
        self.use_reduced_2nd_penalty = Param.use_reduced_2nd_penalty

        self.activation_name = Param.activation_name

        self.nn_features = []
        if Param.nn_features != None :
            feature_str_list =  Param.nn_features.split(';')
            for feature_str in feature_str_list :
                feature_str_split = feature_str.strip("{( )}").split(',')
                # Only diheral angles are considered in the current implementation
                print (feature_str_split)
                if feature_str_split[0] == 'dihedral' : 
                    self.nn_features.append(feature_str_split)

        print ('Features: ', self.nn_features)

        self.train_max_step = Param.train_max_step

        self.load_init_model = Param.load_init_model

        self.init_model_name = Param.init_model_name

        self.include_constraint_step = Param.include_constraint_step

        self.penalty_method = True 

        if self.include_constraint_step == True :
            self.constraint_tol = Param.constraint_tol
            self.constraint_max_step = Param.constraint_max_step
            self.constraint_learning_rate = Param.constraint_learning_rate 
            self.constraint_learning_rate = Param.constraint_learning_rate 
            self.constraint_penalty_method = Param.constraint_penalty_method 
            self.constraint_how_often = Param.constraint_how_often
            self.constraint_first_step = Param.constraint_first_step
            if self.constraint_how_often == 0 :
                self.constraint_how_often = self.train_max_step + 1

        self.print_every_step = Param.print_every_step

        self.print_gradient_norm = Param.print_gradient_norm

        self.data_filename_prefix = Param.data_filename_prefix
        self.eig_file_name_prefix = Param.eig_file_name_prefix
        self.log_filename = Param.log_filename

        # list of (i,j) pairs
        #self.ij_list = list(itertools.combinations_with_replacement(range(Param.k), 2))
        self.ij_list = list(itertools.combinations(range(Param.k), 2))
        self.num_ij_pairs = len(self.ij_list)

        # Set the seed of random number generator 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.set_printoptions(precision=20)

        # Determine the dimension from datafile 
        states_file_name = './data/%s.txt' % (self.data_filename_prefix)
        fp = open(states_file_name, 'r')

        # Only read the first line, in order to find dimension
        K_total, self.dim = [int (x) for x in fp.readline().split()]

        fp.close()

        if Param.namd_data_flag == True : 
            # Compute diagnal matrix based on mass and damping coefficient
            mass_vec = np.loadtxt('./data/namd_mass.txt', skiprows=1)

            assert len(mass_vec) == self.dim, "length of mass array (%d) does not match the dimension (%d)!" % (len(mass_vec), self.dim)

            # This is the friction coefficients, multiplied by the constant 418.4 * 1000, 
            # such that the unit of eigenvalues given by Rayleigh quotients is ns^{-1}.
            self.diag_coeff = torch.from_numpy(418.4 * 1e3 / Param.damping_coeff * np.reciprocal(mass_vec))
        else :
            self.diag_coeff = torch.ones(self.dim).double()

        if self.use_Rayleigh_quotient == False and self.use_reduced_2nd_penalty == True :
            self.use_reduced_2nd_penalty = False 
            print ('Change the flag use_reduced_2nd_penalty=False, since reduced 2nd-order penalty is supported only when Rayleigh quotient is used.\n')

        print ("Diagonal constants (size=%d):\n" % len(self.diag_coeff), self.diag_coeff)
        print ("[Info]  beta = %.4f" % (self.beta))
        print ("[Info]  seed = %d" % (seed))
        print ("[Info]  dim = %d\n" % self.dim)
        print ('\tStages: ', self.stage_list)
        print ('\tBatch size list: ', self.batch_size_list)
        print ('\tLearning rate list: ', self.learning_rate_list)

        if self.all_k_eigs == True : 
            print ("[Info] Compute the first %d eigenvalues" % (self.k))
            print ("[Info] Weights =", self.eig_w)
            if len(self.eig_w) < self.k : 
                print ('Error: only %d weights are provided for %d eigenvalues' % (len(self.eig_w), self.k))
                sys.exit(1)
            if any(x <= 0 for x in self.eig_w) :
                print ('Error: weights in w must be positive!')
                sys.exit(1)
            if any (self.eig_w[i] <= self.eig_w[i+1] for i in range(self.k-1)):
                print ('Error: weights in w are not strictly descending!')
                sys.exit(1)
        else :
            print ("[Info] Compute the %dth eigenvalue" % (self.k))


    # Load sampled data from text file
    def load_data_from_text_file(self):

        # Reads states from file 
        states_file_name = './data/%s.txt' % (self.data_filename_prefix)
        state_weight_vec = np.loadtxt(states_file_name, skiprows=1)
        
        # Number of states 
        K = state_weight_vec.shape[0]

        print ("[Info] loaded data from file: %s\n\t%d states" % (states_file_name, K), flush=True)

        # The last column of the data contains the weights
        return state_weight_vec[:,:-1], state_weight_vec[:,-1]

    def zero_model_parameters(self, model):
        for param in model.parameters():
            torch.nn.init.zeros_(param)
            param.requires_grad=False

    def copy_model_to_bak(self, cvec):
        for i in range(self.k):
            idx = cvec[i]
            for param_tensor in self.model.nets[idx].state_dict():
                self.model_bak.nets[i].state_dict()[param_tensor].copy_(self.model.nets[idx].state_dict()[param_tensor].detach())

    def record_model_parameters(self, des, src, cvec):
        if self.all_k_eigs : 
            for i in range(self.k):
                idx = cvec[i]
                for param_tensor in src.nets[idx].state_dict():
                    des.nets[i].state_dict()[param_tensor].add_(src.nets[idx].state_dict()[param_tensor].detach())
        else :
            for param_tensor in src.state_dict():
                des.state_dict()[param_tensor].add_(src.state_dict()[param_tensor].detach())

    # Perform the following operations on coefficients of neural networks :
    # 1) Compute mean and variance of averaged model on full sample data
    # 2) Shift mean value, 
    # 3) Normalize 
    def update_mean_and_var_of_model(self, model):

        # Evaluate function value on full data
        y = model(self.X_vec).detach()

        # Total weights, will be used for normalization 
        b_tot_weights = self.weights.sum()

        # Mean and variance evaluated on data
        mean_of_nn = [(y[:,idx] * self.weights).sum() / b_tot_weights for idx in range(self.k)]
        var_of_nn = [(y[:,idx]**2 * self.weights).sum() / b_tot_weights - mean_of_nn[idx]**2 for idx in range(self.k)]

        # Step 2 and 3
        model.shift_and_normalize(mean_of_nn, var_of_nn) 


    """
      Update the learning rate of optimizer, when a new training stage starts. 
    """
    def lr_scheduler(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    """ 
      Evaluate functions and their spatial gradients on (local) batch data 
    """
    def fun_and_grad_on_data(self, batch_size):

        # Randomly generate indices of samples from data set 
        x_batch_index = random.choices(range(self.K), cum_weights=self.cum_weights, k=batch_size)
        #x_batch_index = range(self.K)

        #  Choose samples corresonding to those indices,
        #  and reshape the array to avoid the problem when dim=1
        x_batch = torch.reshape(self.X_vec[x_batch_index], (batch_size, self.dim))

        # This is needed in order to compute spatial gradients
        x_batch.requires_grad = True

        # Evaluate function value on data
        self.y = self.model(x_batch)

        # Vector used in computing spatial gradient of functions in pytorch, 
        # where each component of the vector equals 1.
        v_in_jac = torch.ones(batch_size, dtype=torch.float64)

        """
          Apply the Jacobian-vector trick to compute spatial gradients.
          The flag create_graph=True is needed, because later we need to compute
          gradients w.r.t. parameters; Please refer to the torch.autograd.grad function for details.
        """
        self.y_grad_vec = [torch.autograd.grad(self.y[:,idx], x_batch, v_in_jac, create_graph=True)[0] for idx in range(self.k)]

        if self.batch_uniform_weight == True : 
            self.b_weights= self.weights[x_batch_index]
        else :
            self.b_weights= torch.ones(batch_size, dtype=torch.float64)

        # Total weights, will be used for normalization 
        self.b_tot_weights = self.b_weights.sum()

        # Mean and variance evaluated on data
        self.mean_list = [(self.y[:,idx] * self.b_weights).sum() / self.b_tot_weights for idx in range(self.k)]
        self.var_list = [(self.y[:,idx]**2 * self.b_weights).sum() / self.b_tot_weights - self.mean_list[idx]**2 for idx in range(self.k)]

        """
        self.b_angles = self.angles[x_batch_index,:]

        c7eq_index = reduce(np.logical_and, (self.b_angles[:,0] > -120, self.b_angles[:,0] < -70, self.b_angles[:,1] > 40, self.b_angles[:,1] < 130))
        #c7eq_index = self.b_angles[:,0] < 0
        c7ax_index = reduce(np.logical_and, (self.b_angles[:, 0] > 40, self.b_angles[:, 0] < 100, self.b_angles[:,1] < 0, self.b_angles[:,1] > -150))
        l_trans_index = reduce(np.logical_and, (self.b_angles[:,0] > -25, self.b_angles[:,0] < 25, self.b_angles[:,1] > -110, self.b_angles[:,1] < 0))
        r_trans_index = reduce(np.logical_and, (self.b_angles[:,0] > 115, self.b_angles[:,0] < 140, self.b_angles[:,1] > -170, self.b_angles[:,1] < -30))

        self.set_of_index_set = [c7eq_index, c7ax_index, l_trans_index, r_trans_index]

        print ('Name\tnum\t min-w\tmax-w\tsum-w\ttot-w\tpercent')

        num_of_regions = 4
        region_names = ['C7eq', 'C7ax', 'l-channel', 'r-channel']

        for idx_set in  range(num_of_regions) :
            index_set = self.set_of_index_set[idx_set] 
            core_weight = self.b_weights[index_set].sum()
            if core_weight > 0 :
                print ('%8s: %6d\t%8.2e\t%.2e\t%.2e\t%.1f\t%.2e' %
                    (region_names[idx_set], len(self.b_weights[index_set]),
                        self.b_weights[index_set].min(),
                        self.b_weights[index_set].max(), core_weight,
                        self.b_weights.sum(), core_weight / self.b_weights.sum() ) )

        self.neg_index = x_batch[:,0] < 0
        self.pos_index = x_batch[:,0] > 0
        self.trans_index = np.logical_and(x_batch[:,0] > -0.3, x_batch[:,0] < 0.3).type(torch.bool)

        print (x_batch[self.neg_index].shape[0], x_batch[self.trans_index].shape[0], x_batch[self.pos_index].shape[0])
        print (sum(self.b_weights[self.neg_index]), sum(self.b_weights[self.trans_index]), sum(self.b_weights[self.pos_index]))
        print ('neg states: %d\n sum weight: %.4f (total: %.4f, ratio: %.4f)\n \
                mean: %.4f (total: %.4f)\nvar:%.4f (total: %4f) \n' % (x_batch[neg_index].shape[0],
                    sum(self.b_weights[neg_index]), self.b_tot_weights,
                    sum(self.b_weights[neg_index]) / self.b_tot_weights,
                    (self.y[neg_index,0] * self.b_weights[neg_index]).sum() / self.b_tot_weights, 
                self.mean_list[0],
                (self.y[neg_index,0]**2 * self.b_weights[neg_index]).sum() / self.b_tot_weights, 
        self.var_list[0]) )
        """


    # Penalty terms corresonding to 1st-order and 2nd-order constraints
    def penalty_terms(self) :

        penalty = torch.zeros(2, requires_grad=True).double()

        if self.use_Rayleigh_quotient == False :
          # Include mean constraint in penalty
          penalty[0] = sum([self.mean_list[idx]**2 for idx in range(self.k)])

        if self.use_reduced_2nd_penalty == False :
          # Sum of squares of variance for each eigenfunction
          penalty[1] = sum([(self.var_list[idx] - 1.0)**2 for idx in range(self.k)])

          for idx in range(self.num_ij_pairs):
            ij = self.ij_list[idx]
            # Sum of squares of covariance between two different eigenfunctions
            penalty[1] += ((self.y[:, ij[0]] * self.y[:, ij[1]] * self.b_weights).sum() / self.b_tot_weights - self.mean_list[ij[0]] * self.mean_list[ij[1]])**2
        else : 
          for idx in range(self.num_ij_pairs) :
            ij = self.ij_list[idx]
            # Sum of squares of normalized covariance (or correlation coefficient) between two different eigenfunctions
            penalty[1] += ((self.y[:, ij[0]] * self.y[:, ij[1]] * self.b_weights).sum() / self.b_tot_weights - self.mean_list[ij[0]] * self.mean_list[ij[1]])**2 / (self.var_list[ij[0]] * self.var_list[ij[1]])

        return penalty 

    # Constraint steps: train neural networks in order to meet constraints
    # (i.e. mean, variance, orthogonality of eigenfunctions)
    def constraint_update_step(self, bsz):

        constraint_step_num = 0 
        flag = True

        while flag :

            # Compute function values and spatial gradients on batch data
            self.fun_and_grad_on_data(bsz)

            # Compute penalties
            penalty = self.penalty_terms()

            loss = penalty[0] + penalty[1]

            # Update training parameters according to minimize penalty loss
            self.constraint_optimizer.zero_grad()
            loss.backward()
            self.constraint_optimizer.step()

            constraint_step_num += 1 

            if loss < self.constraint_tol : # Success 
                flag = False
            if constraint_step_num % 10 == 0 :  # Print information
               print('Constraint steps:%d,   constraints= [%.4e, %.4e]' % (constraint_step_num, penalty[0], penalty[1]), flush=True)  
            if constraint_step_num >= self.constraint_max_step : # Failed
                print("Constraint tolerance %.4f not reached in %d steps.\n" % (self.constraint_tol, constraint_step_num), flush=True)  
                exit(1)

        print('Total constraint steps: %d,   constraints= [%.4e, %.4e]' % (constraint_step_num, penalty[0], penalty[1]), flush=True)  

    """
      This function calculates the loss function, 
      and updates the neural network functions according to its gradient.

      The loss function consists of 
           (1) either (when computing the kth eigenvalue)
                  the largest eigenvalue of certain k by k matrix , 
               or (when computing the first k eigenvalues)
                  the linear combination (weighted by eig_w) of k Energies or Rayleigh quotients; 
           (2) k 1st-constraints (precisely, weighted inner products wrt. constant function);
           (3) k 2nd-constraints (i.e. orthogonality and normalization constraints). 

      alpha_vec:  two penalty constants used in Steps (2)-(3) above.


      Unit of eigenvalues for NAMD applications:
        length:             angstrom, 10^{-10}m ;
        damping cofficient: ps^{-1} = 10^{12} s^{-1} ;
        mass:               10^{-3} kg/mol, (atomic mass unit: M_u) ;
        1/beta:             kcal/mol (conversion: 1 kcal/mol=4184 J/mol, 1J=1kg*m^2*s^{-2}) ;

        As a result, the unit of Rayleigh quotient is 
           1kcal/mol * 10^{20} m^{-2} * 10^{-12} s / (10^{-3} kg/mol) 
         = 418.4 * ps^{-1}

        Based on this calculation, we include the constant 418.4 in the diag_coeff, 
        so that the eigenvalues are measured in ps^{-1}.

    """
    def update_step(self, bsz, alpha_vec):

        # Compute function values and spatial gradients on batch data
        self.fun_and_grad_on_data(bsz)

        if self.all_k_eigs == False :
            """
              In this case, the min-max principle for the kth eigenvalue is used as the loss function;
              We need to compute the largest eigenvalue and eigenvector of certain kxk matrix
            """

            pair_energy = np.empty((self.k, self.k))

            # Loop over all the pair of indices (i,j) 
            for idx in range(self.num_ij_pairs):
                ij = self.ij_list[idx]

                # Entries of the matrix: weighted averages (scaled by 1/beta) of the inner product of two gradients
                pair_energy[ij[0]][ij[1]] = 1.0 / (self.b_tot_weights * self.beta) * torch.sum((self.y_grad_vec[ij[0]] * self.y_grad_vec[ij[1]] * self.diag_coeff).sum(dim=1) * self.b_weights)

                # Assign the rest entries by symmetry 
                if ij[0] != ij[1] : 
                    pair_energy[ij[1]][ij[0]] = pair_energy[ij[0]][ij[1]] 

            # Compute the eigenpairs 
            vals, vecs = LA.eig(pair_energy)

            # Find out the index of the largest eigenvalue, since the array may not be sorted 
            max_idx = np.argmax(vals)

            # The corresonding eigenvector gives the maximizer c in (the inner 'max' part of) the min-max principle
            cvec = torch.tensor(vecs[:, max_idx])

            # Compute linear combination of \sum_{i=1}^k c_if_i, and its gradient
            cy = torch.matmul(self.y, cvec)
            cy_grad = sum([cvec[j1] * self.y_grad_vec[j1] for j1 in range(k)])

            # Since we impose constraints by penalty terms, \sum_{i=1}^k c_if_i is not exactly normalized. 
            # Here, we use its norm when computing the eigenvalue (but not in the loss function)
            cy_norm = math.sqrt( (cy**2).mean() )
            eig_vals = torch.tensor(vals[max_idx] / cy_norm)

            """
               Compute the partial loss.
               In this case, we simply use energy instead of the Rayleigh quotient.

               Note that in the loss function we ignore the dependance of the
               maximizer c on the functions (f_i)_{1\le i \le k} (therefore on the training parameters)!  
            """
            non_penalty_loss = 1.0 / (self.b_tot_weights * beta) * torch.sum((cy_grad**2 * self.diag_coeff).sum(dim=1) * self.b_weights)
        else :
            # In this case we compute the first k eigenvalues.

            # Always Rayleigh quotients when estimating eigenvalues
            eig_vals = torch.tensor([1.0 / (self.b_tot_weights * self.beta) * torch.sum((self.y_grad_vec[idx]**2 * self.diag_coeff).sum(dim=1) * self.b_weights) / self.var_list[idx] for idx in range(self.k)])

            cvec = range(self.k)
            if self.sort_eigvals_in_training :
                cvec = np.argsort(eig_vals)
                # Sort the eigenvalues 
                eig_vals = eig_vals[cvec]

            # The loss function is the linear combination of k terms.
            if self.use_Rayleigh_quotient == False :
                # Use energies
                non_penalty_loss = 1.0 / (self.b_tot_weights * self.beta) * sum([self.eig_w[idx] * torch.sum((self.y_grad_vec[cvec[idx]]**2 * self.diag_coeff).sum(dim=1) * self.b_weights) for idx in range(self.k)])
            else :
                # Use Rayleigh quotients (i.e. energy divided by variance)
                non_penalty_loss = 1.0 / (self.b_tot_weights * self.beta) * sum([self.eig_w[idx] * torch.sum((self.y_grad_vec[cvec[idx]]**2 * self.diag_coeff).sum(dim=1) * self.b_weights) / self.var_list[cvec[idx]]  for idx in range(self.k)])

                """
                partial_loss = []
                partial_loss_noweight = []
                for index_set in self.set_of_index_set :
                    num = np.sum(index_set)
                    neg_non_penalty_loss = 1.0 / (self.b_tot_weights * self.beta) * \
                    sum([self.eig_w[idx] * torch.sum((self.y_grad_vec[cvec[idx]][index_set,:]**2 * self.diag_coeff).sum(dim=1) * self.b_weights[index_set]) / self.var_list[cvec[idx]]  for idx in range(self.k)])
                    neg_non_penalty_loss_noweight = sum([self.eig_w[idx] *
                        torch.sum((self.y_grad_vec[cvec[idx]][index_set,:]**2).sum(dim=1))
                        / self.var_list[cvec[idx]]  for idx in range(self.k)]) / num

                    partial_loss = partial_loss + [neg_non_penalty_loss.detach().item()] 
                    partial_loss_noweight = partial_loss_noweight + [neg_non_penalty_loss_noweight.detach().item()]

                print ('partial loss: ', partial_loss )
                print ('partial loss noweight: ', partial_loss_noweight )
                """

        # Always compute penalty terms, even if not used
        penalty = self.penalty_terms()

        # Total loss 
        if self.penalty_method == True :
            loss = 1.0 * non_penalty_loss + alpha_vec[0] * penalty[0] + alpha_vec[1] * penalty[1]
        else :
            # In this case, constraint is solved in constraint_update_step. 
            # Therefore, in this step, train neuron network without penalty
            loss = non_penalty_loss

        # Update training parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return eig_vals.numpy(), cvec.numpy(), loss, non_penalty_loss, penalty

    def train(self):

        # Set the current training stage to zero
        stage_index = 0

        log_f = open('./data/%s' % self.log_filename, 'w')

        # Loop over train steps
        for i in range(self.train_max_step) :

            # Update training parameters when a new training stage starts
            if i == self.stage_list[stage_index] :

                # New batch size 
                bsz = self.batch_size_list[stage_index] 

                # Update learning rate
                self.lr_scheduler(self.learning_rate_list[stage_index])

                # Penalty constants 
                alpha_vec = [self.alpha_1_list[stage_index], self.alpha_2_list[stage_index]] 

                # Initialize mean and variance of eigenvalues in this stage
                if self.all_k_eigs : 
                    mean_eig_vals = np.zeros(self.k)
                    var_eig_vals = np.zeros(self.k)
                else :
                    mean_kth_eig = 0.0
                    var_kth_eig = 0.0
                    averaged_cvec = np.zeros(self.k)

                # Reset parameters of averaged_model to zero
                self.zero_model_parameters(self.averaged_model)

                print ('\n[Info] Start %dth training stage from step %d\n\t batch size=%d, lr=%.4f, alpha=[%.2f,%.2f]\n' % (stage_index+1, i, bsz, self.learning_rate_list[stage_index], alpha_vec[0], alpha_vec[1]))

                # Update the current stage index
                stage_index += 1 

            if self.include_constraint_step == True and i >= self.constraint_first_step :
                if self.constraint_penalty_method == False :
                    self.penalty_method = False
                if i % self.constraint_how_often == 0 :
                    # Train neural networks to meet constraints 
                    self.constraint_update_step(bsz)

            # Train neuron networks to minimize loss 
            eig_vals, cvec, loss, non_penalty_loss, penalty = self.update_step(bsz, alpha_vec)
            #eig_vals, cvec, loss, non_penalty_loss, penalty = self.test_update_step(bsz, alpha_vec)

            # Update the statistics of eigenvalues
            if self.all_k_eigs :
                for ii in range(self.k) :
                    mean_eig_vals[ii] += eig_vals[ii]
                    var_eig_vals[ii] += eig_vals[ii]**2
            else :
                mean_kth_eig += eig_vals
                var_kth_eig += eig_vals**2
                averaged_cvec += cvec 

            self.record_model_parameters(self.averaged_model, self.model, cvec)

            # Print information, if we are at the end of current stage or in the last step
            if i + 1 == self.stage_list[stage_index] or i + 1 == self.train_max_step :

                # Compute total number of steps in this stage
                tot_step_in_stage = i + 1 - self.stage_list[stage_index - 1]

                print ('\nStage %d, Step %d to %d (total %d steps):' % (stage_index, self.stage_list[stage_index - 1], i, tot_step_in_stage) )

                if self.all_k_eigs == False :
                    mean_kth_eig /= tot_step_in_stage
                    var_kth_eig = var_kth_eig / tot_step_in_stage - mean_kth_eig**2
                    print ('  %dth eig: mean=%.4f, var=%.4f' % (self.k, mean_kth_eig, math.sqrt(var_kth_eig)) )
                else :
                    for ii in range(self.k) :
                        mean_eig_vals[ii] /= tot_step_in_stage
                        var_eig_vals[ii] = var_eig_vals[ii] / tot_step_in_stage - mean_eig_vals[ii]**2
                        print ('  %dth eig:  mean=%.4f, var=%.4f' % (ii+1, mean_eig_vals[ii], math.sqrt(var_eig_vals[ii])) )

                self.zero_model_parameters(self.model_bak)
                self.copy_model_to_bak(cvec)
                self.update_mean_and_var_of_model(self.model_bak)
                # Save networks to file 
                file_name = './data/%s_stage%d.pt' % (self.eig_file_name_prefix, stage_index)
                torch.save(self.model_bak, file_name)

                # Take average of previous steps
                for param in self.averaged_model.parameters():
                    param /= tot_step_in_stage
                self.update_mean_and_var_of_model(self.averaged_model)
                # Save networks to file 
                file_name = './data/%s_stage%d_averaged.pt' % (self.eig_file_name_prefix, stage_index)
                torch.save(self.averaged_model, file_name)

                if self.all_k_eigs == False :
                    # Also save the c vector to file
                    averaged_cvec /= tot_step_in_stage
                    file_name = './data/cvec_stage%d.txt' % (stage_index)
                    np.savetxt(file_name, averaged_cvec, header='%d' % (self.k), comments="", fmt="%.10f")

            # Display some training information
            if i % self.print_every_step == 0 :
                print( '\ni=%d, stage %d' % (i, stage_index)) 
                print( '   loss= %.4e' % (loss) )
                if self.all_k_eigs == False :
                    print('   %dth eigenvalue = %.4f' % (self.k, eig_vals) )
                else :
                    print('   eigenvalues= ', eig_vals)
                print('   constraints= [%.4e, %.4e]' % (penalty[0], penalty[1]), flush=True)  

                # Print the vector or matrix norm of the gradient.
                if self.print_gradient_norm == True : 
                    grad_list = np.concatenate([p.grad.numpy().flatten()  for p in self.model.parameters()], axis=None)
                    coeff_list = np.concatenate([p.data.numpy().flatten() for p in self.model.parameters()], axis=None)
                    print('   range of parameters: [%.4f, %.4f]' % (min(coeff_list), max(coeff_list)), flush=True)
                    print('   range of gradients: [%.4f, %.4f]' % (min(grad_list), max(grad_list)), flush=True)

                elapsed_time = time.process_time() - self.start_time
                print( '   runtime: %.2f Sec' % elapsed_time )

                # Store the log info
                log_f.write('%d ' % i)
                log_info_vec = [loss, non_penalty_loss, penalty[0], penalty[1]]
                if self.all_k_eigs == False : 
                    log_info_vec.append(eig_vals)
                else :
                    log_info_vec.extend(eig_vals)

                np.savetxt(log_f, np.asarray(log_info_vec).reshape(1,-1), fmt="%.6f")
                log_f.flush()

        log_f.close()

    # Call this function to train networks
    def run(self) :

        # Starting time
        self.start_time = time.process_time()

        # Load trajectory data (states and their weights)
        self.X_vec, self.weights = self.load_data_from_text_file()

        # Reads angles of states from file 
        angle_file_name = './data/angle_along_traj.txt'
        self.angles = np.loadtxt(angle_file_name, skiprows=1)

        if self.weights.min() <= -1e-8 :
            print ( 'Error: weights of states can not be negative (min=%.4e)!' % (self.weights.min()) )
            sys.exit()

        # Size of the trajectory data
        self.K = self.X_vec.shape[0]

        if self.batch_uniform_weight == True : 
            self.cum_weights = np.arange(1, self.K+1)
        else :
            self.cum_weights = np.cumsum(self.weights)

        # Convert to torch type 
        self.X_vec = torch.from_numpy(self.X_vec)
        self.weights = torch.from_numpy(self.weights)

        if len(self.nn_features) > 0 :
            # Include the feature layer and the output layers of neural network
            self.arch_list = [2 * len(self.nn_features)] + self.arch_list + [1]
        else :
            # Include the input/output layers of neural network
            self.arch_list = [self.dim] + self.arch_list + [1]


        # Load trained neural network
        if self.load_init_model == True :
            print( '\n[Info] Load init model from: %s\n' % self.init_model_name )
            file_name = './data/%s' % (self.init_model_name)
            self.model = torch.load(file_name)
            self.model.train()
        else :
            # Initialize networks 
            self.model = network_arch.MyNet(self.arch_list, self.activation_name, self.k, self.nn_features) 
        self.model_bak = network_arch.MyNet(self.arch_list, self.activation_name, self.k, self.nn_features)

        # These networks record training results of several consecutive training steps
        self.averaged_model = network_arch.MyNet(self.arch_list, self.activation_name, self.k, self.nn_features)

        # Use double precision
        self.model.double()
        self.model_bak.double()
        self.averaged_model.double()

        # Initialize Adam optimizier, with initial learning rate for stage 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_list[0])

        # Create another optimizer for solving constriants
        if self.include_constraint_step == True :
            self.constraint_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.constraint_learning_rate)

        # Display some information 
        tot_num_parameters = sum([p.numel() for p in self.model.parameters()])
        elapsed_time = time.process_time() - self.start_time

        print( '\n[Info] Time used in loading data: %.2f Sec\n' % elapsed_time )
        print('\n[Info] Total number of parameters in networks: %d\n' % tot_num_parameters) 
        print ("[Info]  NN architecture:", self.arch_list)
        print ('Range of weights: [%.3e, %.ee]' % (self.weights.min(), self.weights.max()) )

        # Train the networks
        self.train()

        # Output training results
        file_name = './data/%s.pt' % (self.eig_file_name_prefix)
        torch.save(self.model_bak, file_name)

        file_name = './data/%s_averaged.pt' % (self.eig_file_name_prefix)
        torch.save(self.averaged_model, file_name)

        if self.all_k_eigs == False :
            file_name = './data/cvec.txt' 
            np.savetxt(file_name, averaged_cvec, header='%d' % (self.k), comments="", fmt="%.10f")

        elapsed_time = time.process_time() - self.start_time
        print( '\nTotal Runtime: %.2f Sec\n' % elapsed_time )

