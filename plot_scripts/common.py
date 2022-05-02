import configparser

working_dir_list = ['./examples/test-ex1/', './examples/test-ex2']
task_name_list = ['example 1', 'example 2']

task_id = 0
conjugated_eigvec_flag = 0

with_FVD_solution = False
#with_FVD_solution = True

working_dir_name = working_dir_list[task_id]
task_name = task_name_list[task_id]

# read parameters from config file
config = configparser.ConfigParser()
config.read_file(open('../%s/params.cfg' % working_dir_name))

md_flag = config['default'].getboolean('md_data_flag')
num_k = config['Training'].getint('eig_k')
eig_file_name_prefix = config['default'].get('eig_file_name_prefix')
log_filename = config['default'].get('log_filename')

if md_flag:
    data_filename_prefix = config['MD'].get('data_filename_prefix')
    data_filename_prefix_validation = config['MD'].get('data_filename_prefix_validation')
else :
    dim = config['SDE'].getint('dim')
    data_filename_prefix = config['SDE'].get('data_filename_prefix')


