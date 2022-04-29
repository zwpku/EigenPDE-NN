#!/usr/bin/env python3

# Compute a two-dimensional eigenvalue problem using finite volume method

import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
from pylab import *
from numpy import linalg as LA

import potentials 
import read_parameters 

# Form the matrix of the (conjugate) generator 
def build_matrx():
    A = PETSc.Mat().create()
    A.setSizes([nx * ny, nx*ny])
    A.setFromOptions()
    A.setUp()

    Istart, Iend=A.getOwnershipRange()
    xx = np.zeros(2)
    for Ii in range(Istart, Iend):
        s = 0
        i = Ii % nx 
        j = Ii // nx 
        xx[0] = (i + 0.5) * dx + xmin
        xx[1] = (j + 0.5) * dy + ymin
        vi = PotClass.V(xx.reshape((1,2)))
        if i > 0 : 
          JJ = j * nx + i - 1 
          xx[0] = i * dx + xmin
          xx[1] = (j + 0.5) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val = exp(-beta * (v_tmp - vi)) / (beta * dx * dx) 
          s += val 
          xx[0] = (i-0.5) * dx + xmin
          xx[1] = (j + 0.5) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val *= exp(-0.5 * beta * (vi - v_tmp))  
          A[Ii,JJ]=val
        if i < nx-1 :
          JJ = j * nx + i + 1
          xx[0] = (i + 1.0) * dx + xmin
          xx[1] = (j + 0.5) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val = exp(-beta * (v_tmp - vi)) / (beta * dx * dx)  
          s += val 
          xx[0] = (i + 1.5) * dx + xmin
          xx[1] = (j + 0.5) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val *= exp(-0.5 * beta * (vi - v_tmp) )  
          A[Ii,JJ]=val
        if j > 0 : 
          JJ = (j - 1) * nx + i 
          xx[0] = (i + 0.5) * dx + xmin
          xx[1] = j * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val = exp(-beta * (v_tmp - vi)) / (beta * dy * dy) 
          s += val 
          xx[0] = (i + 0.5) * dx + xmin
          xx[1] = (j - 0.5) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val *= exp(-0.5 * beta * (vi - v_tmp) )  
          A[Ii,JJ]=val
        if j < ny-1: 
          JJ = (j + 1) * nx + i 
          xx[0] = (i + 0.5) * dx + xmin
          xx[1] = (j + 1.0) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val = exp(-beta * (v_tmp - vi)) / (beta * dy * dy) 
          s += val 
          xx[0] = (i + 0.5) * dx + xmin
          xx[1] = (j + 1.5) * dy + ymin
          v_tmp = PotClass.V(xx.reshape((1,2)))
          val *= exp(-0.5 * beta * (vi - v_tmp) )  
          A[Ii,JJ]=val

        s *= -1 
        A[Ii,Ii]=s
    A.assemble()
    return A

def out_eig_vec(vv, idx):
  # Scatter the vec to the rank 0
  ctx, vout = PETSc.Scatter().toZero(vv) 
  ctx.scatterBegin(vv, vout, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
  ctx.scatterEnd(vv, vout, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
  v_ptr = vout.getArray() 

  if LA.norm(v_ptr) > 1e-8 :
      v_ptr /= (LA.norm(v_ptr) * math.sqrt(dx * dy))

  file_name = "./data/%s_FVD_%d_conjugated.txt" % (eig_file_name_prefix, idx)
  if rank == 0:
      np.savetxt(file_name, v_ptr.reshape(ny,nx), header="%f %f %d\n%f %f %d" % (xmin, xmax, nx, ymin, ymax, ny), comments="", fmt="%.6e")
      print ("%dth eigenvector is stored in file:\n\t%s" % (idx, file_name))

  x_axis = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx)
  y_axis = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ny)
  xvec = np.transpose([np.tile(x_axis, len(y_axis)), np.repeat(y_axis, len(x_axis))])
  weights = exp(0.5 * beta * PotClass.V(xvec))
  v_ptr *= weights

  if LA.norm(v_ptr) > 1e-8 :
      v_ptr /= (LA.norm(v_ptr) * math.sqrt(dx * dy))

  file_name = "./data/%s_FVD_%d.txt" % (eig_file_name_prefix, idx)
  if rank == 0:
      np.savetxt(file_name, v_ptr.reshape(ny,nx), header="%f %f %d\n%f %f %d" % (xmin, xmax, nx, ymin, ymax, ny), comments="", fmt="%.6e")
      print ("\t%s" % (file_name))

#---------- Start of Main Function ------------

Print=PETSc.Sys.Print

opts=PETSc.Options()
rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)
size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
Print ("rank = %d, size=%d" % (rank, size) )

Param = read_parameters.Param(use_sections={'grid', 'training', 'FVD2d'})
PotClass = potentials.PotClass(Param.dim, Param.pot_id, Param.stiff_eps)

dim = Param.dim
# Dimension should be 2

if (dim == None) or (dim != 2) : 
    Print ("Error: dim (dimension) must to be defined as 2 in: params.cfg")
    sys.exit()
else: 
    Print ("dim=%d" % dim)

beta = Param.beta
eig_file_name_prefix = Param.eig_file_name_prefix

# Grid in R^2
xmin = Param.xmin
xmax = Param.xmax
nx = Param.nx
ymin = Param.ymin
ymax = Param.ymax
ny = Param.ny

# Mesh size
dx = (xmax - xmin) / nx
dy = (ymax - ymin) / ny

# The following calls are standard steps
Print ("\nStart building matrix...")
A = build_matrx()
Print ("Building matrix done.")
E=SLEPc.EPS()
E.create()
E.setOperators(A)
E.setProblemType(SLEPc.EPS.ProblemType.HEP)
# Compute one eigenpair more than required
E.setDimensions(Param.k + 1)
E.setWhichEigenpairs(E.Which.LARGEST_REAL)
E.setTolerances(Param.error_tol, Param.iter_n)
E.setConvergenceTest(E.Conv.ABS)
E.setFromOptions()
E.solve()

its=E.getIterationNumber()
Print("No. of iterations=%d" % its)
eps_type = E.getType()
Print("Method:%s" % eps_type)
nev, ncv, mpd = E.getDimensions()
Print ("No. of requested eigenvalues: %d" % nev)
tol, maxit = E.getTolerances()
Print ("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
nconv = E.getConverged()

if nconv > 0:
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    er = vi.copy()
    for i in range(nconv):
        k = E.getEigenpair(i, vr, vi)
        vtmp = vr.copy()
        vtmp.scale(-k.real)
        A.multAdd(vr, vtmp, er)
        error = E.computeError(i)
        Print ("(%12f, %.4f)\t%12f %12f" % (k.real, k.imag, error, er.norm()))

    for i in range(nconv):
        k = E.getEigenpair(i, vr, vi)
        out_eig_vec(vr, i)

