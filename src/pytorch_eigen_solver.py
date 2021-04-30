import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
import torch
from collections import OrderedDict
import time
from numpy import linalg as LA
from itertools import combinations_with_replacement

import network_arch 

class eigen_solver():

    def __init__(self, Param, Comm, seed=3214):
        # The meanings of parameters are commented in the file: read_parameters.py 
        self.k = Param.k
        self.all_k_eigs = Param.all_k_eigs
        self.eig_w = Param.eig_w

        # When MD data is used, compute beta from temperature
        if Param.namd_data_flag == True : 
            self.beta = Param.namd_beta
        else :
            self.beta = Param.beta

        self.stage_list = Param.stage_list
        self.batch_size_list = Param.batch_size_list
        self.sort_eigvals_in_training = Param.sort_eigvals_in_training

        self.learning_rate_list = Param.learning_rate_list
        self.alpha_1_list = Param.alpha_1_list
        self.alpha_2_list = Param.alpha_2_list
        self.arch_list = Param.inner_arch_size_list + [1]

        self.use_Rayleigh_quotient = Param.use_Rayleigh_quotient
        self.ReLU_flag = Param.ReLU_flag

        self.train_max_step = Param.train_max_step
        self.print_every_step = Param.print_every_step

        self.print_gradient_norm = Param.print_gradient_norm

        self.data_filename_prefix = Param.data_filename_prefix
        self.eig_file_name_prefix = Param.eig_file_name_prefix
        self.log_filename = Param.log_filename

        # list of (i,j) pairs
        self.ij_list = list(combinations_with_replacement(range(Param.k), 2))
        self.num_ij_pairs = len(self.ij_list)

        self.distributed_training = Param.distributed_training
        self.distribute_data = Param.distribute_data

        self.dist = Comm.dist
        self.rank = Comm.rank
        self.size = Comm.size

        # Set the seed of random number generator 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.rank == 0 :
            print ("[Info]  beta = %.4f" % (self.beta))
            print ("[Info]  NN architecture:", self.arch_list)
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
        fp = open(states_file_name, 'r')

        # The first line contains the total number of states
        K_total, = [int (x) for x in fp.readline().split()]
        if self.rank == 0 :
            print ("[Info] ")
            print ("[Info] load data from file: %s\n\t%d states" % (states_file_name, K_total), flush=True)

        tmp_list = []
        line_idx = 0

        # Avoid calling fp.readlines(), as it will read in the entire file and may lead to 
        # memory issue when the file is large. 
        # lines = fp.readlines()

        # Go through lines 
        for line in fp :
            # When the flag is true, data will be distributed among processors.
            if self.distribute_data == False or line_idx % self.size == self.rank :
                state = [float (x) for x in line.split(' ')]
                tmp_list.append(state)
            line_idx += 1 

        state_weight_vec = np.array(tmp_list)

        # Number of states 
        K = state_weight_vec.shape[0]

        if self.distributed_training :
            # Synchronization among processors
            ts = torch.zeros(1)
            if self.rank > 0 :
               self.dist.recv(tensor=ts, src=self.rank-1)
            print('\trank=%d, %d sample data loaded' % (self.rank, K), flush=True)
            if self.rank < self.size - 1 :
               self.dist.send(tensor=ts, dst=self.rank+1)
            self.dist.barrier()
        else :
            print('\trank=%d, %d sample data loaded' % (self.rank, K), flush=True)

        # The last column of the data contains the weights
        return state_weight_vec[:,:-1], state_weight_vec[:,-1]

    def zero_model_parameters(self, model):
        for param in model.parameters():
            torch.nn.init.zeros_(param)
            param.requires_grad=False

    def record_model_parameters(self, des, src, cvec):
        if self.all_k_eigs : 
            for i in range(self.k):
                idx = cvec[i]
                for param_tensor in src.nets[idx].state_dict():
                    des.nets[i].state_dict()[param_tensor].add_(src.nets[idx].state_dict()[param_tensor].detach())
        else :
            for param_tensor in src.state_dict():
                des.state_dict()[param_tensor].add_(src.state_dict()[param_tensor].detach())

    def get_mean_from_sum_of_model(self, num):
        for param in self.averaged_model.parameters():
            param /= num

    """ 
      Model parameter (or gradient) averaging;
      This function is only called when distributed_training=True
    """
    def average_model_parameters(self, average_grad=False):
        if average_grad == True :
            for param in self.model.parameters():
                self.dist.all_reduce(param.grad, op=self.dist.ReduceOp.SUM)
                param.grad /= self.size
        else :
            for param in self.model.parameters():
                self.dist.all_reduce(param.data, op=self.dist.ReduceOp.SUM)
                param.data /= self.size

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
        x_batch_index = random.sample(range(self.K), batch_size)

        #  Choose samples corresonding to those indices,
        #  and reshape the array to avoid the problem when dim=1
        x_batch = torch.reshape(self.X_vec[x_batch_index], (batch_size, self.dim))

        # This is needed in order to compute spatial gradients
        x_batch.requires_grad = True

        # Evaluate function value on data
        y = self.model(x_batch)

        # Vector used in computing spatial gradient of functions in pytorch, 
        # where each component of the vector equals 1.
        v_in_jac = torch.ones(batch_size, dtype=torch.float64)

        """
          Apply the Jacobian-vector trick to compute spatial gradients.
          The flag create_graph=True is needed, because later we need to compute
          gradients w.r.t. parameters; Please refer to the torch.autograd.grad function for details.
        """
        y_grad_vec = [torch.autograd.grad(y[:,idx], x_batch, v_in_jac, create_graph=True)[0] for idx in range(self.k)]

        return y, y_grad_vec, self.weights[x_batch_index]

    """
      This function calculates the loss function, 
      and updates the neural network functions according to its gradient.

      When distributed_training=True, each processor computes gradients of the loss function using local data. 
      Then, the gradients are averaged among processors.

      The loss function consists of 
           (1) either (when computing the kth eigenvalue)
                  the largest eigenvalue of certain k by k matrix , 
               or (when computing the first k eigenvalues)
                  the linear combination (weighted by eig_w) of k Energies or Rayleigh quotients; 
           (2) k 1st-constraints (precisely, weighted inner products wrt. constant function);
           (3) k 2nd-constraints (i.e. orthogonality and normalization constraints). 

      alpha_vec:  two penalty constants used in Steps (2)-(3) above.
    """
    def update_step(self, bsz, alpha_vec):

        # Compute function values and spatial gradients on batch data
        y, y_grad_vec, b_weights = self.fun_and_grad_on_data(bsz)

        # Total weights, will be used for normalization 
        b_tot_weights = b_weights.sum()

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
                pair_energy[ij[0]][ij[1]] = 1.0 / (b_tot_weights * self.beta) * torch.sum((y_grad_vec[ij[0]] * y_grad_vec[ij[1]]).sum(dim=1) * b_weights)

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
            cy = torch.matmul(y, cvec)
            cy_grad = sum([cvec[j1] * y_grad_vec[j1] for j1 in range(k)])

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
            non_penalty_loss = 1.0 / (b_tot_weights * beta) * torch.sum((cy_grad**2).sum(dim=1) * b_weights)

        else :
            # In this case we compute the first k eigenvalues.

            # Always Rayleigh quotients when estimating eigenvalues
            eig_vals = torch.tensor([1.0 / self.beta * (torch.sum((y_grad_vec[idx]**2).sum(dim=1) * b_weights) / torch.sum(y[:,idx]**2 * b_weights)) for idx in range(self.k)])

            cvec = range(self.k)
            if self.sort_eigvals_in_training :
                if self.distributed_training : 
                    self.dist.all_reduce(eig_vals, op=self.dist.ReduceOp.SUM)
                    eig_vals /= self.size
                cvec = np.argsort(eig_vals)
                # Sort the eigenvalues 
                eig_vals = eig_vals[cvec]
        
            # The loss function is the linear combination of k terms.
            if self.use_Rayleigh_quotient == False :
                # Use energies
                non_penalty_loss = 1.0 / (b_tot_weights * self.beta) * sum([self.eig_w[idx] * torch.sum((y_grad_vec[cvec[idx]]**2).sum(dim=1) * b_weights) for idx in range(self.k)])
            else :
                # Use Rayleigh quotients (i.e. energy divided by 2nd moments)
                non_penalty_loss = 1.0 / self.beta * sum([self.eig_w[idx] * torch.sum((y_grad_vec[cvec[idx]]**2).sum(dim=1) * b_weights) / torch.sum((y[:,cvec[idx]]**2 * b_weights)) for idx in range(self.k)])

        # Penalty terms corresonding to k 1st-order and k 2nd-order constraints
        penalty = torch.zeros(2, requires_grad=True).double()

        # Sum of squares of 1st-order constraints
        penalty[0] = sum([((y[:,idx] * b_weights).sum() / b_tot_weights)**2 for idx in range(self.k)])

        penalty[1] = 0.0 
        for idx in range(self.num_ij_pairs):
            ij = self.ij_list[idx]
            # Sum of squares of 2nd-order constraints
            penalty[1] += ((y[:, ij[0]] * y[:, ij[1]] * b_weights).sum() / b_tot_weights - int(ij[0] == ij[1]))**2

        # Total loss 
        loss = non_penalty_loss + alpha_vec[0] * penalty[0] + alpha_vec[1] * penalty[1]

        # Update training parameters
        self.optimizer.zero_grad()
        loss.backward()

        if self.distributed_training :
            # Average gradients and other quantities among processors
            self.average_model_parameters(average_grad=True)

            self.dist.all_reduce(loss, op=self.dist.ReduceOp.SUM)
            loss /= self.size 

            self.dist.all_reduce(non_penalty_loss, op=self.dist.ReduceOp.SUM)
            non_penalty_loss /= self.size 

            self.dist.all_reduce(penalty, op=self.dist.ReduceOp.SUM)
            penalty /= self.size

            self.dist.all_reduce(eig_vals, op=self.dist.ReduceOp.SUM)
            eig_vals /= self.size

            if self.all_k_eigs == False :
                self.dist.all_reduce(cvec, op=self.dist.ReduceOp.SUM)
                cvec /= self.size

        self.optimizer.step()

        return eig_vals.numpy(), cvec.numpy(), loss, non_penalty_loss, penalty

    def train(self):

        # Set the current training stage to zero
        stage_index = 0

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

                # Distribute batch size to each processor
                if self.distributed_training :
                    bsz_local = bsz // self.size
                    if self.rank < bsz % self.size :
                       bsz_local += 1

                if self.rank == 0 :
                    print ('\n[Info] Start %dth training stage from step %d\n\t batch size=%d, lr=%.4f, alpha=[%.2f,%.2f]\n' % (stage_index+1, i, bsz, self.learning_rate_list[stage_index], alpha_vec[0], alpha_vec[1]))
                # Update the current stage index

                stage_index += 1 
     
            # Update training parameters 
            if self.distributed_training :
                # Make sure the neuron networks have the same parameters on each processor.
                self.average_model_parameters()

                eig_vals, cvec, loss, non_penalty_loss, penalty = self.update_step(bsz_local, alpha_vec)

            else :
                eig_vals, cvec, loss, non_penalty_loss, penalty = self.update_step(bsz, alpha_vec)

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

            # Print information, if we are at the end of each stage 
            if self.rank == 0 and i + 1 == self.stage_list[stage_index] :

                # Compute total number of steps in this stage
                tot_step_in_stage = i + 1 - self.stage_list[stage_index - 1]

                print ('\nStage %d, Step %d to %d (total %d):' % (stage_index, self.stage_list[stage_index - 1], i, tot_step_in_stage) )

                if self.all_k_eigs == False :
                    mean_kth_eig /= tot_step_in_stage
                    var_kth_eig = var_kth_eig / tot_step_in_stage - mean_kth_eig**2
                    print ('  %dth eig: mean=%.4f, var=%.4f' % (self.k, mean_kth_eig, math.sqrt(var_kth_eig)) )
                else :
                    for ii in range(self.k) :
                        mean_eig_vals[ii] /= tot_step_in_stage
                        var_eig_vals[ii] = var_eig_vals[ii] / tot_step_in_stage - mean_eig_vals[ii]**2
                        print ('  %dth eig:  mean=%.4f, var=%.4f' % (ii+1, mean_eig_vals[ii], math.sqrt(var_eig_vals[ii])) )
                
                self.get_mean_from_sum_of_model(tot_step_in_stage)

                # Save networks to file for each stage
                file_name = './data/%s_stage%d.pt' % (self.eig_file_name_prefix, stage_index)
                torch.save(self.averaged_model, file_name)
                print( '  Neuron neworks at stage %d are saved to %s' % (stage_index, file_name) )

                if self.all_k_eigs == False :
                    # Also save the c vector to file
                    averaged_cvec /= tot_step_in_stage
                    file_name = './data/cvec_stage%d.txt' % (stage_index)
                    np.savetxt(file_name, averaged_cvec, header='%d' % (self.k), comments="", fmt="%.10f")

            # Display some training information
            if self.rank == 0 and i % self.print_every_step == 0 :
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
                if self.log_p_index < self.log_max_n :
                    self.log_info_vec[self.log_p_index, 0:4] = [loss, non_penalty_loss, penalty[0], penalty[1]]
                    if self.all_k_eigs == False : 
                        self.log_info_vec[self.log_p_index, 4] = eig_vals
                    else :
                        self.log_info_vec[self.log_p_index, 4:] = eig_vals
                    self.log_p_index += 1

    # Call this function to train networks
    def run(self) :

        # Initialize networks 
        self.model = network_arch.MyNet(self.arch_list, self.ReLU_flag, self.k)

        # These networks record training results of several consecutive training steps
        self.averaged_model = network_arch.MyNet(self.arch_list, self.ReLU_flag, self.k)

        # Use double precision
        self.model.double()
        self.averaged_model.double()

        # Initialize Adam optimizier, with initial learning rate for stage 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_list[0])

        # Maximal number of items of log data
        self.log_max_n = self.train_max_step // self.print_every_step + 1

        """
          Initialize array for log data, which includes
            (1) total loss; 
            (2) non-penalty part of loss; 
            (3)-(4) two penalty terms; 
            (5)- eigenvalues 
        """
        if self.all_k_eigs == False :
            self.log_info_vec = np.zeros((self.log_max_n, 5))
        else :
            self.log_info_vec = np.zeros((self.log_max_n, 4 + self.k))

        self.log_p_index = 0

        # Starting time
        self.start_time = time.process_time()

        # Load trajectory data (states and their weights)
        self.X_vec, self.weights = self.load_data_from_text_file()

        # Convert to torch type 
        self.X_vec = torch.from_numpy(self.X_vec)
        self.weights = torch.from_numpy(self.weights)

        if self.weights.min() <= -1e-8 :
            if self.rank == 0 :
                print ( 'Error: rank=%d, weights of states can not be negative (min=%.4e)!' % (self.rank, self.weights.min()) )
            sys.exit()

        # Size of the trajectory data
        self.K = self.X_vec.shape[0]

        # Decide the dimension of the data
        if len(self.X_vec.shape) > 1: 
            self.dim = self.X_vec.shape[1]
        else :
            self.dim = 1 

        # Display some information 
        if self.rank == 0:
            tot_num_parameters = sum([p.numel() for p in self.model.parameters()])
            elapsed_time = time.process_time() - self.start_time

            print( '\n[Info] Time used in loading data: %.2f Sec\n' % elapsed_time )
            print ("[Info] dim=%d\n" % self.dim)
            print('\n[Info] Total number of parameters in networks: %d\n' % tot_num_parameters) 

        if self.distributed_training :  
            # On each processor, print information of data size 
            ts = torch.zeros(1)
            if self.rank > 0 :
               self.dist.recv(tensor=ts, src=self.rank-1)
            print ("On rank %d, data size K=%d,\tRange of weights: [%.3e, %.3e]" % (self.rank, self.K, self.weights.min(), self.weights.max()), flush=True)
            if self.rank < self.size - 1 :
               self.dist.send(tensor=ts, dst=self.rank+1)
            self.dist.barrier()
        else :
            print ('Range of weights: [%.3e, %.ee]' % (self.weights.min(), self.weights.max()) )

        # Train the networks
        self.train()

        # Output training results
        if self.rank == 0 :

            file_name = './data/%s.pt' % (self.eig_file_name_prefix)
            torch.save(self.averaged_model, file_name)
            print( '\nNeuron neworks after training are saved to %s' % (file_name) )

            if self.all_k_eigs == False :
                file_name = './data/cvec.txt' 
                np.savetxt(file_name, averaged_cvec, header='%d' % (self.k), comments="", fmt="%.10f")

            elapsed_time = time.process_time() - self.start_time
            print( '\nTotal Runtime: %.2f Sec\n' % elapsed_time )

            np.savetxt('./data/%s' % self.log_filename, self.log_info_vec[0:self.log_p_index, :], header='%d %d' % (self.log_p_index, self.log_info_vec.shape[1]), comments="", fmt="%.10f")


