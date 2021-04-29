#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import potentials 
import read_parameters 

Param = read_parameters.Param()
PotClass = potentials.PotClass(Param.dim, Param.pot_id, Param.stiff_eps)
dim = Param.dim

if dim == 1 :
    # Grid in R
    xmin = Param.xmin
    xmax = Param.xmax
    nx = Param.nx
    dx = (xmax - xmin) / nx
    xvec = np.linspace(xmin, xmax, nx).reshape(1, -1)
    pot_vec = PotClass.V(xvec) 
    pot_filename = './data/pot.txt'
    np.savetxt(pot_filename, np.reshape(pot_vec, nx), header='%f %f %d\n' % (xmin,xmax,nx), comments="", fmt="%.10f")
    print("Potential is stored to: %s\n" % (pot_filename))

if dim >= 2 :
    # Grid in R^2
    xmin = Param.xmin
    xmax = Param.xmax
    nx = Param.nx
    ymin = Param.ymin
    ymax = Param.ymax
    ny = Param.ny

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    x1_axis = np.linspace(xmin, xmax, nx)
    x2_axis = np.linspace(ymin, ymax, ny)

    x1_vec = np.tile(x1_axis, len(x2_axis)).reshape(nx * ny, 1)
    x2_vec = np.repeat(x2_axis, len(x1_axis)).reshape(nx * ny, 1)

    # When dim>2, the components in the other dimensions are set to zero.
    other_x = np.tile(np.zeros(dim-2), nx* ny).reshape(nx* ny, dim-2)
    x2d = np.concatenate((x1_vec, x2_vec, other_x), axis=1)

    pot_vec = PotClass.V(x2d)

    print ("Range of potential: [%.3f, %.3f]" % (min(pot_vec), max(pot_vec)) )

    pot_filename = './data/pot.txt'
    np.savetxt(pot_filename, np.reshape(pot_vec, (ny, nx)), header='%f %f %d\n%f %f %d' % (xmin,xmax,nx, ymin, ymax, ny), comments="", fmt="%.10f")

    print("Potential V is stored to: %s\n" % (pot_filename))

