#!/usr/bin/python3

from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from common import *

fig = figure()
w = 0.19
h = 0.4
pot_id = 1
tot_min = -1.0
tot_max = 1.0

# read parameters from config file

for i in range(num_k):
  if conjugated_eigvec_flag == 1: 
     data_file = open('../%s/data/%s_FVD_%d_conjugated.txt' % (working_dir_name, eig_file_name_prefix, i+1), 'r')
  else :
     data_file = open('../%s/data/%s_FVD_%d.txt' % (working_dir_name, eig_file_name_prefix, i+1), 'r')
  xmin, xmax, nx  = [ float (x) for x in data_file.readline().split() ]
  ymin, ymax, ny  = [ float (x) for x in data_file.readline().split() ]

  Z = np.loadtxt(data_file)
  #print (Z.min(), Z.max())

  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)

  pos = [0.06 + i * (w + 0.03) , 0.4 , w, h]
  ax = fig.add_axes(pos)
  tot_min = Z.min() 
  tot_max = Z.max() 

  im = ax.imshow( Z , cmap=cm.jet, extent = [xmin, xmax, ymin, ymax], vmin=tot_min , vmax=tot_max , origin='lower', interpolation='none' )
  ax = gca() 

  if i == 0:
    yticks(np.linspace(ymin, ymax, 5))
  else :
    plt.setp(ax.get_yticklabels(), visible=False)

  ax.tick_params(axis='y', labelsize=10)
  ax.tick_params(axis='x', labelsize=10)
  xticks(np.linspace(xmin, xmax, 5))
  ax.set_aspect(1.0)

#labels = [ item.get_text() for item in cbar.ax.get_yticklabels() ]
#cbar.ax.set_yticklabels(['%.2f' % (pow(float(item), 2) + Z.min()) for item in labels] )
cax = fig.add_axes([0.1, 0.28, .76, 0.05])
fig.colorbar(im, cax=cax, orientation='horizontal',cmap=cm.jet)
cax.tick_params(labelsize=10)

if conjugated_eigvec_flag == 1 : 
    fig_name = '../%s/fig/eigvec_FVD_all_conjugated.eps' % (working_dir_name)
else :
    fig_name = '../%s/fig/eigvec_FVD_all.eps' % (working_dir_name)
savefig(fig_name)

print ("output figure: %s" % fig_name)

