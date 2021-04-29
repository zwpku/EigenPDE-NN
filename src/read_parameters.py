import configparser

class Param:
    def __init__(self):
        # Read parameters from config file
        config = configparser.ConfigParser()
        config.read_file(open('params.cfg'))

        self.dim = config['default'].getint('dim')
        self.pot_id = config['default'].getint('pot_id')
        self.stiff_eps = config['potential'].getfloat('stiff_eps')

        # This is the step-size used when sampling data. It is not used in training
        self.delta_t = config['sample_data'].getfloat('delta_t')
        # K: total numbers of states
        self.K = int(config['sample_data'].getfloat('N_state'))
        # Inverse temperature. 
        self.beta = config['default'].getfloat('beta')

        # Prefix of filename for trajectory data 
        self.data_filename_prefix = config['default'].get('data_filename_prefix')

        # Prefix of filename for eigenfunctions
        self.eig_file_name_prefix = config['default'].get('eig_file_name_prefix')

        # Index of the eigenvalues to learn
        self.k = config['default'].getint('eig_idx_k')

        # Whether compute all the first k eigenvalues, compute only the kth eigenvalue
        self.all_k_eigs = config['default'].getboolean('compute_all_k_eigs')

        # Weights in the loss function
        self.eig_w = [float(x) for x in config['default'].get('eig_weight').split(',')]

        # Grid in R^2
        self.xmin = config['grid'].getfloat('xmin')
        self.xmax = config['grid'].getfloat('xmax')
        self.nx = config['grid'].getint('nx')
        self.ymin = config['grid'].getfloat('ymin')
        self.ymax = config['grid'].getfloat('ymax')
        self.ny = config['grid'].getint('ny')

        # Tolerance and maximal iterations 
        self.error_tol = config['FVD2d'].getfloat('error_tol')
        self.iter_n = config['FVD2d'].getint('iter_n')

        # Architectual of neural networks
        self.inner_arch_size_list = [int(x) for x in config['NeuralNetArch'].get('arch_size_list').split(',')]

        # Use ReLu or Tanh as activation functions
        self.ReLU_flag = config['NeuralNetArch'].getboolean('ReLU_flag')

        # If true, the pytorch with mpi is needed in order to run the code
        self.distributed_training = config['Training'].getboolean('distributed_training')

        # If true, the whole data will be divided on each processor. 
        self.distribute_data = config['Training'].getboolean('distribute_data')

        # Total gradient steps
        self.train_max_step = config['Training'].getint('train_max_step')

        #Total batch size 
        self.batch_size_list = [int(x) for x in config['Training'].get('batch_size_list').split(',')] 

        # Train stages 
        self.stage_list = [int(x) for x in config['Training'].get('stage_list').split(',')]

        assert len(self.batch_size_list) == len(self.stage_list), "batch sizes are not set correctly!" 
        assert self.stage_list[0] == 0, "the first stage should start from step 0!"

        self.stage_list.append(self.train_max_step)

        # Learning rate for each training stage
        self.learning_rate_list = [float(x) for x in config['Training'].get('learning_rate_list').split(',')]

        # Penalty constants for each training stage
        self.alpha_1_list = [float(x) for x in config['Training'].get('alpha_1_list').split(',')]
        self.alpha_2_list = [float(x) for x in config['Training'].get('alpha_2_list').split(',')]

        # Use Rayleigh quotient or energy 
        self.use_Rayleigh_quotient = config['Training'].getboolean('use_Rayleigh_quotient')

        # If true, eigenvalues will be sorted ascendingly 
        self.sort_eigvals_in_training = config['Training'].getboolean('sort_eigvals_in_training')

        # Frequency to print information
        self.print_every_step = config['Training'].getint('print_every_step')

        # If true, display norm of gradient during training
        self.print_gradient_norm = config['Training'].getboolean('print_gradient_norm')
        
        # If true, train networks using MD data
        self.namd_data_flag = config['default'].getboolean('namd_data_flag')

        if self.namd_data_flag == True : 
            self.temp_T = config['NAMD'].getfloat('temperature')
            self.pdb_path = config['NAMD'].get('pdb_path')
            self.pdb_prefix = config['NAMD'].get('pdb_prefix')
            self.which_data_to_use = config['NAMD'].get('which_data_to_use')
            self.align_data_flag = config['NAMD'].getboolean('align_data_flag')
            self.namd_data_filename_prefix = config['NAMD'].get('namd_data_filename_prefix')
            self.namd_data_path = config['NAMD'].get('namd_data_path')

            # physical quantities 
            # boltzmann constant (eV/K)
            # 1 kcal/mol = 0.043 eV
            Kb = 8.6173333262145 * 1e-5
            self.namd_beta = 0.043 / (Kb * self.temp_T)
            print ("T=%.2f, beta = %.4f" % (self.temp_T, self.namd_beta))

        # Log file for training 
        self.log_filename = config['default'].get('log_filename')

