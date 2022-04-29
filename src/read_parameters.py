import configparser

class Param:
    #This class holds all required parameters, read from the configuration file `params.cfg`.

    def __init__(self, use_sections=set()):

        # Read parameters from config file
        config = configparser.ConfigParser()
        config.read_file(open('params.cfg'))

        # If true, train networks using MD data
        self.md_data_flag = config['default'].getboolean('md_data_flag')

        if self.md_data_flag == True : 
            self.temp_T = config['MD'].getfloat('temperature')
            self.diffusion_coeff = config['MD'].getfloat('diffusion_coeff')
            self.psf_name = config['MD'].get('psf_name')
            self.which_data_to_use = config['MD'].get('which_data_to_use')
            self.use_biased_data = config['MD'].getboolean('use_biased_data')
            self.weight_threshold_to_remove_states = config['MD'].getfloat('weight_threshold_to_remove_states')
            self.align_data_flag = config['MD'].get('align_data_flag')

            self.md_dcddata_filename_prefix = config['MD'].get('md_dcddata_filename_prefix')

            self.md_dcddata_path = config['MD'].get('md_dcddata_path')
            self.data_filename_prefix = config['MD'].get('data_filename_prefix')

            self.md_dcddata_path_validation = config['MD'].get('md_dcddata_path_validation')
            self.data_filename_prefix_validation = config['MD'].get('data_filename_prefix_validation')
            # Physical quantities 
            # Boltzmann constant (unit: kcal/(mol*K)). We use the same value as in MD.
            Kb = 0.001987191 
            # The unit of the PMF obtained from MD is: kcal/mol
            # At T=300, md_beta = 1.6774
            self.md_beta = 1.0 / (Kb * self.temp_T)
        else :
            self.dim = config['SDE'].getint('dim')
            self.pot_id = config['SDE'].getint('pot_id')
            self.stiff_eps = config['SDE'].getfloat('stiff_eps')

            # This is the step-size used when sampling data. It is not used in training
            self.delta_t = config['SDE'].getfloat('delta_t')
            # K: total numbers of states
            self.K = int(config['SDE'].getfloat('N_state'))
            # Using biased data (reweighting) are used when 
            # it is different from the value of beta.
            self.SDE_beta = config['SDE'].getfloat('SDE_beta')
            # Inverse temperature. 
            self.beta = config['SDE'].getfloat('beta')
            # Prefix of filename for trajectory data 
            self.data_filename_prefix = config['SDE'].get('data_filename_prefix')
            self.data_filename_prefix_validation = config['SDE'].get('data_filename_prefix_validation')

        self.load_data_how_often = int(config['default'].getfloat('load_data_how_often'))
        # Prefix of filename for eigenfunctions
        self.eig_file_name_prefix = config['default'].get('eig_file_name_prefix')

        if 'training' in use_sections:
            # number of eigenvalues to learn
            self.k = config['Training'].getint('eig_k')
            # Weights in the loss function
            self.eig_w = [float(x) for x in config['Training'].get('eig_weight').split(',')]

            # Architectual of neural networks
            self.inner_arch_size_list = [int(x) for x in config['NeuralNetArch'].get('arch_size_list').split(',')]

            # Name of activation function
            self.activation_name = config['NeuralNetArch'].get('activation_name')

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
            self.alpha_list = [float(x) for x in config['Training'].get('alpha_list').split(',')]

            # If true, eigenvalues will be sorted ascendingly 
            self.sort_eigvals_in_training = config['Training'].getboolean('sort_eigvals_in_training')

            # Frequency to print information
            self.print_every_step = config['Training'].getint('print_every_step')

        # Log file for training 
        self.log_filename = config['default'].get('log_filename')

        if 'grid' in use_sections:
            # Grid in R^2
            self.xmin = config['grid'].getfloat('xmin')
            self.xmax = config['grid'].getfloat('xmax')
            self.nx = config['grid'].getint('nx')
            self.ymin = config['grid'].getfloat('ymin')
            self.ymax = config['grid'].getfloat('ymax')
            self.ny = config['grid'].getint('ny')

        if 'FVD2d' in use_sections:
            # Tolerance and maximal iterations 
            self.error_tol = config['FVD2d'].getfloat('error_tol')
            self.iter_n = config['FVD2d'].getint('iter_n')

