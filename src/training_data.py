import namd_loader_dipeptide

class PrepareData() :
    def __init__(self, Param) :
        self.Param = Param

    # Sample data by simulating SDE
    def generate_sample_data(self) :

        import potentials 
        import numpy as np
        import random

        PotClass = potentials.PotClass(self.Param.dim, self.Param.pot_id, self.Param.stiff_eps)

        dim = self.Param.dim
        # Step-size in the numerical scheme
        delta_t = self.Param.delta_t 
        # K: total numbers of states
        K = self.Param.K
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
        X_vec = np.zeros((K, dim+1))
        print_step_interval = int(K / 10)

        print ("Next, generate %d states" % K)
        tot_weight = 0.0
        for i in range(K):
            xi = np.random.randn(dim)
            X0 = X0 - PotClass.grad_V(X0.reshape(1,dim)) * delta_t + np.sqrt(2 * delta_t / SDE_beta) * xi
            X_vec[i, 0:dim] = X0
            # Set weights in the last column
            X_vec[i][dim] = np.exp(-(beta - SDE_beta) * PotClass.V(X0.reshape(1,dim)) )
            tot_weight += X_vec[i][dim]
            if i % print_step_interval == 0:
               print ("%4.1f%% finished." % (i / K * 100), flush=True)
        X_vec[:,dim] /= (tot_weight / K)

        states_file_name = './data/%s.txt' % (data_filename_prefix)
        np.savetxt(states_file_name, X_vec, header='%d %d' % (K, dim), comments="", fmt="%.10f")
        print("\nsampled data are stored to: %s" % states_file_name)

    def prepare_data(self) :
        if self.Param.namd_data_flag == True :
            # use MD data 
            print ("Generate training data from MD data\n")
            namd_loader = namd_loader_dipeptide.namd_data_loader(self.Param, False) 
            namd_loader.save_namd_data_to_txt()
        else :
            # Sample data by simulating SDE

            print ("Generate training data by Euler-Maruyama scheme\n")
            self.generate_sample_data() 

