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
import data_set

class eigen_solver():

    def __init__(self, Param, seed=3214):
        # The meanings of parameters are commented in the file: read_parameters.py 
        self.k = Param.k
        self.eig_w = Param.eig_w
        self.namd_data_flag = Param.namd_data_flag

        # When MD data is used, compute beta from temperature
        if  self.namd_data_flag == True : 
            self.beta = Param.namd_beta
        else :
            self.beta = Param.beta

        self.batch_uniform_weight = Param.batch_uniform_weight
        self.stage_list = Param.stage_list
        self.batch_size_list = Param.batch_size_list

        self.sort_eigvals_in_training = Param.sort_eigvals_in_training

        self.learning_rate_list = Param.learning_rate_list
        self.alpha_list = Param.alpha_list

        # Sizes of inner Layers (without input/output layers)
        self.arch_list = Param.inner_arch_size_list 

        self.activation_name = Param.activation_name

        self.train_max_step = Param.train_max_step

        self.load_init_model = Param.load_init_model

        self.init_model_name = Param.init_model_name

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
        states_filename = './data/%s.txt' % (self.data_filename_prefix)
        fp = open(states_filename, 'r')

        # Read the second line, in order to find dimension
        fp.readline()
        self.tot_dim = len(fp.readline().split()) - 1
        fp.close()

        if self.namd_data_flag == True : 
            # This is the diagnoal matrix; 
            # The unit of eigenvalues given by Rayleigh quotients is ns^{-1}.
            self.diag_coeff = torch.ones(self.tot_dim).double() * Param.diffusion_coeff * 1e7 * self.beta
        else :
            self.diag_coeff = torch.ones(self.tot_dim).double()

        print ("[Info]  beta = %.4f" % (self.beta))
        print ("[Info]  seed = %d" % (seed))
        print ("[Info]  dim = %d\n" % self.tot_dim)
        print ('\tStages: ', self.stage_list)
        print ('\tBatch size list: ', self.batch_size_list)
        print ('\tLearning rate list: ', self.learning_rate_list)

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
        for i in range(self.k):
            idx = cvec[i]
            for param_tensor in src.nets[idx].state_dict():
                des.nets[i].state_dict()[param_tensor].add_(src.nets[idx].state_dict()[param_tensor].detach())

    def update_mean_and_var_of_model(self, model):
    # Perform the following operations on coefficients of neural networks :
    # 1) Compute mean and variance of averaged model on full sample data
    # 2) Shift mean value, 
    # 3) Normalize 

        self.dataset.generate_minibatch(self.dataset.K, False) 

        # Evaluate function value on full data
        y = model(self.dataset).detach()

        weights= self.dataset.weights_minibatch()

        # Total weights, will be used for normalization 
        tot_weights = weights.sum()

        # Mean and variance evaluated on data
        mean_of_nn = [(y[:,idx] * weights).sum() / tot_weights for idx in range(self.k)]
        var_of_nn = [(y[:,idx]**2 * weights).sum() / tot_weights - mean_of_nn[idx]**2 for idx in range(self.k)]

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

        self.dataset.generate_minibatch(batch_size) 

        # Evaluate function value on data
        self.y = self.model(self.dataset)

        """
          Apply the Jacobian-vector trick to compute spatial gradients.
          The flag create_graph=True is needed, because later we need to compute
          gradients w.r.t. parameters; Please refer to the torch.autograd.grad function for details.
        """
        self.y_grad_vec = [torch.autograd.grad(self.y[:,idx], self.dataset.x_batch, self.v_in_jac, create_graph=True)[0] for idx in range(self.k)]

        self.b_weights= self.dataset.weights_minibatch()

        # Total weights, will be used for normalization 
        self.b_tot_weights = self.b_weights.sum()

        # Mean and variance evaluated on data
        self.mean_list = [(self.y[:,idx] * self.b_weights).sum() / self.b_tot_weights for idx in range(self.k)]
        self.var_list = [(self.y[:,idx]**2 * self.b_weights).sum() / self.b_tot_weights - self.mean_list[idx]**2 for idx in range(self.k)]


    # Penalty term corresonding to constraints
    def penalty_term(self) :

      # Sum of squares of variance for each eigenfunction
      penalty = sum([(self.var_list[idx] - 1.0)**2 for idx in range(self.k)])

      for idx in range(self.num_ij_pairs):
        ij = self.ij_list[idx]
        # Sum of squares of covariance between two different eigenfunctions
        penalty += ((self.y[:, ij[0]] * self.y[:, ij[1]] * self.b_weights).sum() / self.b_tot_weights - self.mean_list[ij[0]] * self.mean_list[ij[1]])**2

        return penalty 

    def update_step(self, bsz, alpha_val):
        """
          This function calculates the loss function, 
          and updates the neural network functions according to its gradient.

          The loss function consists of 
               (1) linear combination (weighted by eig_w) of k Rayleigh quotients; 
               (2) second-order constraints (i.e. orthogonality and normalization constraints). 

          alpha_val:  penalty constant used in Step (2) above.

          Unit of eigenvalues for Brownian dynamics: 
            length:                 angstrom, 10^{-10}m ;
            diffusion coefficient:  cm^2 s^{-1} = 10^{-13} m^2 ns^{-1}

            As a result, the unit of Rayleigh quotient is 
               10^{-13} m^2 ns^{-1} * 10^{20} m^{-2} = 10^7 ns^{-1}
            This calculation will be used to compuate the diagnoal matrix.
        """

        # Compute function values and spatial gradients on batch data
        self.fun_and_grad_on_data(bsz)

        # Always Rayleigh quotients when estimating eigenvalues
        eig_vals = torch.tensor([1.0 / (self.b_tot_weights * self.beta) * torch.sum((self.y_grad_vec[idx]**2 * self.diag_coeff).sum(dim=1) * self.b_weights) / self.var_list[idx] for idx in range(self.k)])

        cvec = range(self.k)
        if self.sort_eigvals_in_training :
            cvec = np.argsort(eig_vals)
            # Sort the eigenvalues 
            eig_vals = eig_vals[cvec]

        # Use Rayleigh quotients (i.e. energy divided by variance)
        non_penalty_loss = 1.0 / (self.b_tot_weights * self.beta) * sum([self.eig_w[idx] * torch.sum((self.y_grad_vec[cvec[idx]]**2 * self.diag_coeff).sum(dim=1) * self.b_weights) / self.var_list[cvec[idx]]  for idx in range(self.k)])

        # Always compute penalty terms, even if not used
        penalty = self.penalty_term()

        loss = 1.0 * non_penalty_loss + alpha_val * penalty

        # Update training parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return eig_vals.numpy(), cvec.numpy(), loss.detach().numpy(), non_penalty_loss.detach().numpy(), penalty.detach().numpy()

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
                alpha_val = self.alpha_list[stage_index] 

                # Vector used in computing spatial gradient of functions in pytorch, 
                # where each component of the vector equals 1.
                self.v_in_jac = torch.ones(bsz, dtype=torch.float64)

                # Initialize mean and variance of eigenvalues in this stage
                mean_eig_vals = np.zeros(self.k)
                var_eig_vals = np.zeros(self.k)

                # Reset parameters of averaged_model to zero
                self.zero_model_parameters(self.averaged_model)

                print ('\n[Info] Start %dth training stage from step %d\n\t batch size=%d, lr=%.4f, alpha=%.2f\n' % (stage_index+1, i, bsz, self.learning_rate_list[stage_index], alpha_val))

                # Update the current stage index
                stage_index += 1 

            # Train neuron networks to minimize loss 
            eig_vals, cvec, loss, non_penalty_loss, penalty = self.update_step(bsz, alpha_val)

            # Update the statistics of eigenvalues
            for ii in range(self.k) :
                mean_eig_vals[ii] += eig_vals[ii]
                var_eig_vals[ii] += eig_vals[ii]**2

            self.record_model_parameters(self.averaged_model, self.model, cvec)

            # Print information, if we are at the end of current stage or in the last step
            if i + 1 == self.stage_list[stage_index] or i + 1 == self.train_max_step :

                # Compute total number of steps in this stage
                tot_step_in_stage = i + 1 - self.stage_list[stage_index - 1]

                print ('\nStage %d, Step %d to %d (total %d steps):' % (stage_index, self.stage_list[stage_index - 1], i, tot_step_in_stage) )

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

            # Display some training information
            if i % self.print_every_step == 0 :
                print( '\ni=%d, stage %d' % (i, stage_index)) 
                print( '   loss= %.4e' % (loss) )
                print('   eigenvalues= ', eig_vals)
                print('   constraints= %.4e' % (penalty), flush=True)  

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
                log_info_vec = [loss, non_penalty_loss, penalty]
                log_info_vec.extend(eig_vals)

                np.savetxt(log_f, np.asarray(log_info_vec).reshape(1,-1), fmt="%.6f")
                log_f.flush()

        log_f.close()

    # Call this function to train networks
    def run(self) :

        # Starting time
        self.start_time = time.process_time()

        states_filename = './data/%s.txt' % (self.data_filename_prefix)

        # Load trajectory data 
        if self.namd_data_flag == True :
            self.dataset = data_set.MD_data_set.from_file(states_filename)
        else :
            self.dataset = data_set.data_set.from_file(states_filename)

        if self.batch_uniform_weight == False : 
            self.dataset.set_nonuniform_batch_weight() 

        if self.namd_data_flag == True :
            self.dataset.load_ref_state() 

        # Include the input/output layers of neural network
        self.arch_list = [self.dataset.tot_dim] + self.arch_list + [1]

        # Load trained neural network
        if self.load_init_model == True :
            print( '\n[Info] Load init model from: %s\n' % self.init_model_name )
            file_name = './data/%s' % (self.init_model_name)
            self.model = torch.load(file_name)
            self.model.train()
            # Set explicitly for continuing training
            for param in self.model.parameters():
                param.requires_grad=True

        else :
            # Initialize networks 
            self.model = network_arch.MyNet(self.arch_list, self.activation_name, self.k) 

        self.model_bak = network_arch.MyNet(self.arch_list, self.activation_name, self.k)

        # These networks record training results of several consecutive training steps
        self.averaged_model = network_arch.MyNet(self.arch_list, self.activation_name, self.k)

        # Use double precision
        self.model.double()
        self.model_bak.double()
        self.averaged_model.double()

        # Initialize Adam optimizier, with initial learning rate for stage 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_list[0])

        # Display some information 
        tot_num_parameters = sum([p.numel() for p in self.model.parameters()])
        elapsed_time = time.process_time() - self.start_time

        print( '\n[Info] Time spent for loading data: %.2f Sec' % elapsed_time )
        print( '[Info] Total number of parameters in networks: %d' % tot_num_parameters )  
        print ("[Info]  NN architecture:", self.arch_list)

        # Train the networks
        self.train()

        # Output training results
        file_name = './data/%s.pt' % (self.eig_file_name_prefix)
        torch.save(self.model_bak, file_name)

        file_name = './data/%s_averaged.pt' % (self.eig_file_name_prefix)
        torch.save(self.averaged_model, file_name)

        elapsed_time = time.process_time() - self.start_time
        print( '\nTotal Runtime: %.2f Sec\n' % elapsed_time )

