import configparser

working_dir_list = ['working_dir_namd_ABF7e-1-800ns-MultiFeatures', 'working_dir_1d_beta1.4-rerun', 'working_dir_2d_MetastableRadius_single_1', 'working_dir_namd_ABF1e0_Tanh30-30-30-80ns', 'working_dir_namd_ABF1e0_Tanh30-30-30', 'working_dir_namd_NoABF-nonh', 'working_dir_5d_MetastableRadius', 'working_dir_20d_MetastableRadius', 'working_dir_40d_MetastableRadius_newpot', 'working_dir_100d_MetastableRadius', 'working_dir_100d_MetastableRadius_dist']
task_name_list = ['2D metastable', 'Alanine Dipeptide', '100Dim, metastable in x_1 and x_2', '100Dim, metastable in x_1 and x_2', '100Dim, metastable in x_1 and x_2', '100Dim, metastable in x_1 and x_2']

task_id = 0
conjugated_eigvec_flag = 0

with_FVD_solution = False
#with_FVD_solution = True

working_dir_name = working_dir_list[task_id]
task_name = task_name_list[task_id]

# read parameters from config file
config = configparser.ConfigParser()
config.read_file(open('../%s/params.cfg' % working_dir_name))

dim = config['default'].getint('dim')

data_filename_prefix = config['NAMD'].get('data_filename_prefix')
data_filename_prefix_validation = config['NAMD'].get('data_filename_prefix_validation')

eig_file_name_prefix = config['default'].get('eig_file_name_prefix')
num_k = config['default'].getint('eig_k')

log_filename = config['default'].get('log_filename')


