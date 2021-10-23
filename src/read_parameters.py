import configparser

class Param:

    """
    This class holds all required parameters, read from the configuration file `params.cfg`.

    :ivar pot_id: index of potentials.
    :ivar dim: dimension of the system.
    """
    
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
        self.load_data_how_often = int(config['default'].getfloat('load_data_how_often'))
        # Coefficient under which the data is sampled. 
        # Using biased data (reweighting) are used when 
        # it is different from the value of beta.
        self.SDE_beta = config['sample_data'].getfloat('SDE_beta')

        # Inverse temperature. 
        self.beta = config['default'].getfloat('beta')

        # Prefix of filename for trajectory data 
        self.data_filename_prefix = config['default'].get('data_filename_prefix')

        # Prefix of filename for eigenfunctions
        self.eig_file_name_prefix = config['default'].get('eig_file_name_prefix')

        # number of eigenvalues to learn
        self.k = config['default'].getint('eig_k')

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

        # Name of activation function
        self.activation_name = config['NeuralNetArch'].get('activation_name')

        try:
            self.nn_features = config['NeuralNetArch']['features']
        except:
            self.nn_features = None

        if self.nn_features != None :
            self.write_feature_to_file = config['Training'].getboolean('write_feature_to_file')
        else: 
            self.write_feature_to_file = False

        # Total gradient steps
        self.train_max_step = config['Training'].getint('train_max_step')

        # Whether start from a trained model 
        self.load_init_model = config['Training'].getboolean('load_init_model')

        # the filename of trained model
        self.init_model_name = config['Training'].get('init_model_name')

        #Total batch size 
        self.batch_size_list = [int(x) for x in config['Training'].get('batch_size_list').split(',')] 

        self.batch_uniform_weight = config['Training'].getboolean('batch_uniform_weight')

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

        self.use_reduced_2nd_penalty = config['Training'].getboolean('use_reduced_2nd_penalty')

        # Include extra steps for constriants
        self.include_constraint_step = config['Training'].getboolean('include_constraint_step')
        if self.include_constraint_step == True :
            self.constraint_first_step = config['Training'].getint('constraint_first_step')
            self.constraint_tol = config['Training'].getfloat('constraint_tol')
            self.constraint_learning_rate = config['Training'].getfloat('constraint_learning_rate')
            self.constraint_max_step = config['Training'].getint('constraint_max_step')
            self.constraint_penalty_method  = config['Training'].getboolean('constraint_penalty_method')
            self.constraint_how_often = config['Training'].getint('constraint_how_often')

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
            self.diffusion_coeff = config['NAMD'].getfloat('diffusion_coeff')
            self.psf_name = config['NAMD'].get('psf_name')
            self.which_data_to_use = config['NAMD'].get('which_data_to_use')
            self.use_biased_data = config['NAMD'].getboolean('use_biased_data')
            self.weight_threshold_to_remove_states = config['NAMD'].getfloat('weight_threshold_to_remove_states')
            self.align_data_flag = config['NAMD'].get('align_data_flag')

            self.namd_dcddata_filename_prefix = config['NAMD'].get('namd_dcddata_filename_prefix')

            self.namd_dcddata_path = config['NAMD'].get('namd_dcddata_path')
            self.data_filename_prefix = config['NAMD'].get('data_filename_prefix')

            self.namd_dcddata_path_validation = config['NAMD'].get('namd_dcddata_path_validation')
            self.data_filename_prefix_validation = config['NAMD'].get('data_filename_prefix_validation')
            # Physical quantities 
            # Boltzmann constant (unit: kcal/(mol*K)). We use the same value as in NAMD.
            Kb = 0.001987191 
            # The unit of the PMF obtained from NAMD is: kcal/mol
            # At T=300, namd_beta = 1.6774
            self.namd_beta = 1.0 / (Kb * self.temp_T)

        # Log file for training 
        self.log_filename = config['default'].get('log_filename')


