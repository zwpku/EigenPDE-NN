import configparser

working_dir_list = ['working_dir_2d', 'working_dir_namd_ABF7e-1_RemoveRotation_NN40-80-40', 'working_dir_namd_NoABF-nonh', 'working_dir_namd_abf-full-80ns', 'working_dir_namd_abf-nonh-80ns', 'working_dir_namd_abf-nonh-varying-20ns', 'working_dir_2d_MetastableRadius_test', 'working_dir_5d_MetastableRadius', 'working_dir_20d_MetastableRadius', 'working_dir_40d_MetastableRadius_newpot', 'working_dir_100d_MetastableRadius', 'working_dir_100d_MetastableRadius_dist']
task_name_list = ['2D metastable', 'Alanine Dipeptide', '100Dim, metastable in x_1 and x_2', '100Dim, metastable in x_1 and x_2', '100Dim, metastable in x_1 and x_2', '100Dim, metastable in x_1 and x_2']

task_id = 1
conjugated_eigvec_flag = 0

with_FVD_solution = False
#with_FVD_solution = True

working_dir_name = working_dir_list[task_id]
task_name = task_name_list[task_id]

# read parameters from config file
config = configparser.ConfigParser()
config.read_file(open('../%s/params.cfg' % working_dir_name))

dim = config['default'].getint('dim')
data_filename_prefix = config['default'].get('data_filename_prefix')
eig_file_name_prefix = config['default'].get('eig_file_name_prefix')
num_k = config['default'].getint('eig_idx_k')

log_filename = config['default'].get('log_filename')
all_eig_flag = config['default'].getboolean('compute_all_k_eigs')


