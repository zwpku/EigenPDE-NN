import namd_loader_dipeptide
import potentials 
import numpy as np
import random

class PrepareData() :
    def __init__(self, Param) :
        self.Param = Param

    # Sample data by simulating SDE
    def generate_sample_data(self) :

        PotClass = potentials.PotClass(self.Param.dim, self.Param.pot_id, self.Param.stiff_eps)

        PotClass.output_potential(self.Param)

        print ("Generate training data by Euler-Maruyama scheme\n")
        dim = self.Param.dim
        # Step-size in the numerical scheme
        delta_t = self.Param.delta_t 
        # K: total numbers of states
        K = self.Param.K
        how_often = self.Param.load_data_how_often

        if how_often == 0 : 
            how_often = 1

        # True beta 
        beta = self.Param.beta
        # beta used in sampling 
        SDE_beta = self.Param.SDE_beta

        # Filename of the trajectory data
        data_filename_prefix = self.Param.data_filename_prefix

        print ("beta=%.2f\tdelta_t=%.2e\tSDE_beta=%.2f\n" % (beta, delta_t, SDE_beta))

        X0 = np.random.randn(dim)

        # Number of steps in the initialization phase
        burn_step = 10000
        print ("First, burning, total step = %d" % burn_step)
        for i in range(burn_step):
            xi = np.random.randn(dim)
            X0 = X0 - PotClass.grad_V(X0.reshape(1,dim)) * delta_t + np.sqrt(2 * delta_t / SDE_beta) * xi

        # The last column contains the importance sampling weights of the states
        X_vec = np.zeros((K//how_often, dim+1))
        print_step_interval = int(K / 10)

        print ("Next, generate %d states" % K)
        kk = 0 
        for i in range(K):
            xi = np.random.randn(dim)
            X0 = X0 - PotClass.grad_V(X0.reshape(1,dim)) * delta_t + np.sqrt(2 * delta_t / SDE_beta) * xi

            if i % how_often == 0 :
                X_vec[kk, 0:dim] = X0
                # Set weights in the last column
                X_vec[kk][dim] = np.exp(-(beta - SDE_beta) * PotClass.V(X0.reshape(1,dim)) )
                kk += 1

            if i % print_step_interval == 0:
               print ("%4.1f%% finished." % (i / K * 100), flush=True)

        mean_weight = np.mean(X_vec[:,dim])
        X_vec[:,dim] /= mean_weight 

        states_file_name = './data/%s.txt' % (data_filename_prefix)
        np.savetxt(states_file_name, X_vec[:kk,:], header='%d %d' % (kk, dim), comments="", fmt="%.10f")
        print("\nSampled data are stored to: %s" % states_file_name)

    def prepare_data(self) :
        if self.Param.namd_data_flag == True :
            # use MD data 
            print ("Using MD data as training data\n")
            namd_loader = namd_loader_dipeptide.namd_data_loader(self.Param, False) 
            namd_loader.save_namd_data_to_txt()
        else :
            # Sample data by simulating SDE

            self.generate_sample_data() 

