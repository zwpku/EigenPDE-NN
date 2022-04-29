import md_loader_dipeptide
import potentials 
import numpy as np
import random

class PrepareData() :
    """
    This class prepares trajectory data for training neural networks.

    :param Param: contains all parameters.
    :type Param: :py:class:`read_parameters.Param`
    """

    def __init__(self, Param) :
        self.Param = Param

    def generate_sample_data(self, training_data) :
        """
        Generate sample data by simulating SDE using Euler-Maruyama scheme.

        The data will be stored to file under `./data` directory with the filename
        specified in `Param`.
        """

        PotClass = potentials.PotClass(self.Param.dim, self.Param.pot_id, self.Param.stiff_eps)

        PotClass.output_potential(self.Param)

        if training_data:
            print ("Generate training data by Euler-Maruyama scheme\n")
            data_filename_prefix = self.Param.data_filename_prefix
        else :
            print ("Generate validataion data by Euler-Maruyama scheme\n")
            data_filename_prefix = self.Param.data_filename_prefix_validation

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

    def prepare_data(self, training_data=True) :
        """
        Prepare training/testing data. 

        When `Param.md_data_flag=True`, it loads MD data using the class :class:`md_loader_dipeptide.md_data_loader`; 
        otherwise, it generates sample data by calling
        :py:meth:`generate_sample_data` (the potential in use is specified by :py:data:`read_parameters.Param.pot_id`).

        :param training_data: whether generate data for training or for testing.
        :type training_data: bool

        """
        if self.Param.md_data_flag == True :
            # use MD data 
            print ("Using MD data as training/validation data\n")
            md_loader = md_loader_dipeptide.md_data_loader(self.Param, training_data) 
            md_loader.save_md_data_to_txt()
        else :
            # sample data by simulating SDE

            self.generate_sample_data(training_data) 

