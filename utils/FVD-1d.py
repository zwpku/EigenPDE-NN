#!/usr/bin/env python3

# Compute a one-dimensional eigenvalue problem using finite volume method

from pylab import *
import numpy as np
from numpy import linalg as LA

import sys
sys.path.append('../src/')

import potentials 
import read_parameters

Param = read_parameters.Param()
PotClass = potentials.PotClass(Param.dim, Param.pot_id, Param.stiff_eps)

# Dimension should be one
dim = Param.dim
if (dim == None) or (dim != 1) : 
    print ("Error: dim (dimension) must to be defined to be 1 in: params.cfg")
    sys.exit()

beta = Param.beta
eig_file_name_prefix = Param.eig_file_name_prefix

# Range of grid, [xmin, xmax] ;
# Since it is 1D problem, only use grid along x axis.
xmin = Param.xmin
xmax = Param.xmax
# Number of grids
nx = Param.nx
# Mesh size
dx = (xmax - xmin) / nx

print (nx)

# Compute one eigenvalue more than required
ev_num = Param.k + 1

# Matrix corresonding to discretized generator
mat_a = np.zeros((nx, nx))

# Form the matrix of the generator by finite volume method
for i in range(0, nx):
    x = xmin + (i+0.5) * dx
    if i > 0:
        # Centers of two adjoint cells on the left side
        x0 = xmin + (i-0.5) * dx
        x1 = xmin + i * dx
        mat_a[i][i-1] = -exp( beta * 0.5 * (PotClass.V(np.array(x0).reshape(1,1)) +
            PotClass.V(np.array(x).reshape(1,1)) - 2 * PotClass.V(np.array(x1).reshape(1,1))) ) / (beta * dx**2)
        mat_a[i][i] = exp( beta * (PotClass.V(np.array(x).reshape(1,1)) - PotClass.V(np.array(x1).reshape(1,1)) ) ) / (beta * dx**2)
    if i < nx-1:     
        # Centers of two adjoint cells on the right side
        x0 = xmin + (i+1.5) * dx
        x1 = xmin + (i+1) * dx
        mat_a[i][i+1] = -exp( beta * 0.5 * (PotClass.V(np.array(x0).reshape(1,1)) + PotClass.V(np.array(x).reshape(1,1)) - 2 * PotClass.V( np.array(x1).reshape(1,1))) ) / (beta * dx**2)
        mat_a[i][i] = mat_a[i][i] + exp( beta * (PotClass.V(np.array(x).reshape(1,1)) - PotClass.V(np.array(x1).reshape(1,1)) ) ) / (beta * dx**2)

# Solve the (conjugated) eigenvalue problem.
# This function will return all eigenpairs, in ascending order
w,vv=LA.eigh(mat_a)

print ("the first %d eigenvalues: " % ev_num) 
print (w[0:ev_num])

# Store the eigenvectors  
for idx in range(ev_num):
    eigen_file_name = "./data/%s_FVD_%d_conjugated.txt" % (eig_file_name_prefix, idx)
    # First normalize, then save
    np.savetxt(eigen_file_name, vv[:,idx] / (norm(vv[:,idx]) * math.sqrt(dx)), header='%f %f %d\n' % (xmin,xmax,nx), comments="", fmt="%.10f")
    print("\n%dth eigen function is stored to:\n\t%s" % (idx, eigen_file_name))

    # Transform back the (unconjugated) eigenvectors 
    yvec = [(exp(0.5*beta * PotClass.V(np.array(xmin + (i+0.5)*dx).reshape(1,1))) * vv[i][idx]) for i in range(nx)]
    eigen_file_name = "./data/%s_FVD_%d.txt" % (eig_file_name_prefix, idx)
    # Normalize, then save
    np.savetxt(eigen_file_name, yvec / (norm(yvec) * math.sqrt(dx)), header='%f %f %d\n' % (xmin,xmax,nx), comments="", fmt="%.10f")
    print("\t%s" % (eigen_file_name))

