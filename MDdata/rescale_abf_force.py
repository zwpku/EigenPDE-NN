import matplotlib.pyplot as plt
import numpy as np
import shutil

def rescale_force (filename, filename_new, alpha):
    fp = open(filename)
    out_fp = open(filename_new, "w") 
    for i in range(3):
        out_fp.writelines(fp.readline())

    angle_data = np.loadtxt(filename)
    angle_data[:,2] *= alpha
    angle_data[:,3] *= alpha
    np.savetxt(out_fp, angle_data, fmt='%.2f\t%.2f\t%.7f\t%.7f') 

data_path = './abf-varying-20ns/'
new_data_path = './abf-fixed-100ns-0.7/'
filename_prefix = 'colvarso'
scale_cst = 0.7

colvar_force_filenanme = '%s/%s.grad' % (data_path, filename_prefix)
colvar_count_filename = '%s/%s.count' % (data_path, filename_prefix)

new_colvar_force_filenanme = '%s/%s-new.grad' % (new_data_path, filename_prefix)
new_colvar_count_filename = '%s/%s-new.count' % (new_data_path, filename_prefix)

rescale_force(colvar_force_filenanme, new_colvar_force_filenanme, scale_cst)

print ('rescale the gradients in file:\t %s,\nsave data in new file:\t%s\nalpha=%.2f' % (colvar_force_filenanme, new_colvar_force_filenanme, scale_cst) )

shutil.copy2(colvar_count_filename, new_colvar_count_filename)

print ('copy count file: %s\nto new file: %s' % (colvar_count_filename, new_colvar_count_filename))
