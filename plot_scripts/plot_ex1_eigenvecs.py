#!/usr/bin/python3

from pylab import *
from numpy import *
import matplotlib.cm as cm

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

num_k = 3
fig, ax = plt.subplots(num_k, 4, figsize=(9, 16), gridspec_kw={'hspace': -0.75, 'wspace': 0.35})

working_dir_list = ['working_dir_2d_MetastableRadius', 'working_dir_50d_MetastableRadius', 'working_dir_100d_MetastableRadius']

#fig.suptitle('Eigenfunctions, %s' % task_name)
working_dir_name = working_dir_list[0]
eig_file_name_prefix = 'eigen_vector'
tot_min = -0.3
tot_max = 0.3

for i in range(3) :
      data_file = open('../%s/data/%s_FVD_%d.txt' % (working_dir_name, eig_file_name_prefix, i+1), 'r')

      xmin, xmax, nx = [ float (x) for x in data_file.readline().split() ]
      ymin, ymax, ny = [ float (x) for x in data_file.readline().split() ]

      Z = np.loadtxt(data_file)
      x = np.linspace(xmin, xmax, nx)
      y = np.linspace(ymin, ymax, ny)
#      tot_min = Z.min() tot_max = Z.max() 
      im = ax[i, 0].imshow(Z, cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )
      yticks(np.linspace(xmin, xmax, 5))
      ax[i,0].set_ylabel(r'$\varphi_%d$' % (i+1),fontsize=26, labelpad=-1, rotation=0)
      ax[i,0].yaxis.set_label_coords(-0.35, 0.42)

ax[0,0].set_title(r'FVM, $d=2$', fontsize=24)

sign_list = [-1, 1, 1]

working_dir_name = working_dir_list[0]
eig_file_name_prefix = 'eigen_vector'

for i in range(num_k) : 
  data_file = open('../%s/data/%s_%d.txt' % (working_dir_name, eig_file_name_prefix, i+1), 'r')

  xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
  ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]

  Z = np.loadtxt(data_file, skiprows=0)

  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(x,y)

  #tot_min = Z.min() tot_max = Z.max() 

  im = ax[i,1].imshow(sign_list[i] * Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )
  #plt.setp(ax[i,1].get_yticklabels(), visible=False)

ax[0,1].set_title(r'NN, $d=2$',  fontsize=24)

working_dir_name = working_dir_list[1]
eig_file_name_prefix = 'eigen_vector'
sign_list = [1, 1, 1]

for i in range(num_k) : 
  data_file = open('../%s/data/%s_%d.txt' % (working_dir_name, eig_file_name_prefix, i+1), 'r')

  xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
  ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]

  Z = np.loadtxt(data_file, skiprows=0)

  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(x,y)

  #tot_min = Z.min() tot_max = Z.max() 
#  print (tot_min, tot_max)

  im = ax[i,2].imshow( sign_list[i] * Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )
  #plt.setp(ax[i,2].get_yticklabels(), visible=False)

ax[0,2].set_title(r'NN, $d=50$', fontsize=24)

working_dir_name = working_dir_list[2]
eig_file_name_prefix = 'eigen_vector'
sign_list = [-1, 1, 1]

for i in range(num_k) : 
  data_file = open('../%s/data/%s_%d.txt' % (working_dir_name, eig_file_name_prefix, i+1), 'r')

  xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
  ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]

  Z = np.loadtxt(data_file, skiprows=0)

  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(x,y)

  #tot_min = Z.min() tot_max = Z.max() 
#  print (tot_min, tot_max)

  im = ax[i,3].imshow( sign_list[i] * Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )
  #plt.setp(ax[i,3].get_yticklabels(), visible=False)

ax[0,3].set_title(r'NN, $d=100$', fontsize=24)


for i in range (4) :
    for j in range(3):
        ax[j,i].set_xticks([-2,0,2])
        ax[j,i].set_yticks([-2,0,2])
        ax[j,i].tick_params(axis='y', labelsize=20, pad=1.5)
        ax[j,i].tick_params(axis='x', labelsize=20)

cax = fig.add_axes([0.93, 0.32, 0.04, 0.35])
#fig.colorbar(im, cax=cax, orientation='horizontal',cmap=cm.jet)
fig.colorbar(im, cax=cax, cmap=cm.jet)
cax.tick_params(labelsize=20)

fig_name = './ex1_eigvec_nn_and_fvd.eps' 
savefig(fig_name)
fig.savefig(fig_name, bbox_inches='tight')
print ("output figure: %s" % fig_name)

