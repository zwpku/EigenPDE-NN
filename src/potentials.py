import numpy as np
import math 
from math import pi

class PotClass():
    def __init__(self, dim, pot_id, stiff_eps):
        self.dim = dim
        self.stiff_eps = stiff_eps 
        self.pot_id = pot_id

    # f(x)-cx, where c>0, f(0)=b, f(a)=f(-a)=0, and f(x)->\infty quadratically
    def modified_dw1d(self, xvec, a=1.0, b=1.0, c=0.6):
        y = np.zeros(len(xvec))
        for idx in range(len(xvec)) :
            x = xvec[idx]
            if x > a :
              y[idx] = b * pi**2 / (4 * a**2) * (x - a) ** 2
            if x < -a :
              y[idx] = b * pi**2 / (4 * a**2) * (x + a) ** 2
            if x >= -a and x <= a :
              y[idx] = b * 0.5 * (math.cos(x / a * pi) + 1.0)
            y[idx] -= c * x
        return y

    def grad_modified_dw1d(self, xvec, a=1.0, b=1.0, c=0.6): 
        y = np.zeros(len(xvec))
        for idx in range(len(xvec)) :
            x = xvec[idx]
            if x > a :
                y[idx] = b * pi**2 / (2 * a**2) * (x - a)
            if x < -a :
                y[idx] = b * pi**2 / (2 * a**2) * (x + a)
            if x >= -a and x <= a : 
                y[idx] = - b * pi / (2 * a) * math.sin(x * pi / a)
            y[idx] -= c
        return y.reshape((-1, 1))

    # pot_id=1
    # 1d quadratic
    def v_1d_quadratic(self, x):
        return 0.5 * x ** 2 
    def grad_v_1d_quadratic(self, x): 
        return x

    # pot_id=2
    # 1d double well
    def v_1d_dw(self, x):
        return self.modified_dw1d(x, b=1) 
    def grad_v_1d_dw(self, x): 
        return self.grad_modified_dw1d(x,b=1)

    # pot_id=3 
    # 2d quadratic
    def v_2d_quadratic(self, x):
        return 0.5 * x[:,0]**2 + 2.0 * x[:,1]**2
    def grad_v_2d_quaratic(self, x): 
        return np.column_stack((x[:,0], 4.0 * x[:,1]))

    # pot_id=4
    # 2d, double well in x0, Gaussian in x1
    def v_2d_dw_quadratic(self, x):
        return self.modified_dw1d(x[:,0]) + 0.5 * x[:,1]**2
    def grad_v_2d_dw_quadratic(self, x): 
        return np.column_stack((grad_modified_dw1d(x[:,0]),x[:,1]))

    # pot_id=5
    # 2d, three wells along radius
    def v_2d_3well(self, x):
      # angle in [-pi, pi] 
      theta = np.arctan2(x[:,1], x[:,0])
      # radius
      r = np.sqrt( x[:,0] * x[:,0] + x[:,1] * x[:,1] )

      v_vec = np.zeros(len(x))
      for idx in range(len(x)) :
          # potential V_1
          if theta[idx] > pi / 3 : 
            v_vec[idx] = (1-(theta[idx] * 3 / pi- 1.0)**2)**2
          if theta[idx] < - pi / 3 : 
            v_vec[idx] = (1-(theta[idx] * 3 / pi + 1.0)**2)**2
          if theta[idx] > -pi / 3 and theta[idx] < pi / 3:
            v_vec[idx] = 3.0 / 5.0 - 2.0 / 5.0 * np.cos(3 * theta[idx])  
      # potential V_2
      v_vec = v_vec * 1.0 + (r - 1)**2 * 1.0 / self.stiff_eps + 5.0 * np.exp(-5.0 * r**2) 
      return v_vec

    def grad_v_2d_3well(self, x): 
      # angle
      theta = np.arctan2(x[:,1], x[:,0])
      # radius
      r = np.sqrt( x[:,0] * x[:,0] + x[:,1] * x[:,1] )

      if any(np.fabs(r) < 1e-8): 
          print ("warning: radius is too small! r=%.4e" % r)
      dv1_dangle = np.zeros(len(x))
      # derivative of V_1 w.r.t. angle
      for idx in range(len(x)) :
          if theta[idx] > pi / 3: 
            dv1_dangle[idx] = 12 / pi * (theta[idx] * 3 / pi - 1) * ((theta[idx] * 3 / pi- 1.0)**2-1)
          if theta[idx] < - pi / 3: 
            dv1_dangle[idx] = 12 / pi * (theta[idx] * 3 / pi + 1) * ((theta[idx] * 3 / pi + 1.0)**2-1)
          if theta[idx] > -pi / 3 and theta[idx] < pi / 3:
            dv1_dangle[idx] = 1.2 * math.sin (3 * theta[idx])
      # derivative of V_2 w.r.t. angle
      dv2_dangle = np.zeros(len(x))
      # derivative of V_2 w.r.t. radius
      dv2_dr = 2.0 * (r-1.0) / self.stiff_eps - 50.0 * r * np.exp(-r**2/0.2)

      return np.column_stack((-(dv1_dangle + dv2_dangle) * x[:,1] / (r * r)+ dv2_dr * x[:,0] / r,  (dv1_dangle + dv2_dangle) * x[:,0] / (r * r)+ dv2_dr * x[:,1] / r))

    #pot_id = 6
    def v_2d_curved(self, x):
        return (x[:,0]**2 - 1.0)**2 + 1.0 / self.stiff_eps * (x[:,0]**2 + x[:,1] - 1.0)**2 

    def grad_v_2d_curved(self, x):
        return np.column_stack((4.0 * (x[:,0]**2 - 1.0) * x[:,0] + 4.0 / self.stiff_eps * (x[:,0]**2 + x[:,1] - 1.0) * x[:,0], 2.0 / self.stiff_eps * (x[:,0]**2 + x[:,1] - 1.0)))

    #pot_id = 7
    #high dim, 3 well in x[0] and x[1], quadratic in other dimensions
    def v_nd_3well_in_x01(self, x):
        return self.v_2d_3well(x[:, 0:2]) + sum([0.5 * 10.0 * x[:,i]**2 for i in range(2,self.dim)])
    def grad_v_nd_3well_in_x01(self, x):
        return np.concatenate((self.grad_v_2d_3well(x[:,0:2]), np.array([1.0 * 10.0 * x[:,i] for i in range(2,self.dim)]).T), axis=1)


    def V(self, x):
        if self.pot_id == 1 :
            return self.v_1d_quadratic(x)
        if self.pot_id == 2 :
            return self.v_1d_dw(x)
        if self.pot_id == 3 :
            return self.v_2d_quadratic(x)
        if self.pot_id == 4 :
            return self.v_2d_dw_quadratic(x)
        if self.pot_id == 5 :
            return self.v_2d_3well(x)
        if self.pot_id == 6 :
            return self.v_2d_curved(x)
        if self.pot_id == 7 :
            return self.v_nd_3well_in_x01(x)

    def grad_V(self, x_vec):
        if self.pot_id == 1 :
            return self.grad_v_1d_quadratic(x_vec)
        if self.pot_id == 2 :
            return self.grad_v_1d_dw(x_vec)
        if self.pot_id == 3 :
            return self.grad_v_2d_quaratic(x_vec) 
        if self.pot_id == 4 :
            return self.grad_v_2d_dw_quadratic(x_vec) 
        if self.pot_id == 5 :
            return self.grad_v_2d_3well(x_vec) 
        if self.pot_id == 6 :
            return self.grad_v_2d_curved(x_vec)
        if self.pot_id == 7 :
            return self.grad_v_nd_3well_in_x01(x_vec)

    def output_potential(self, Param) :
        if self.dim == 1 :
            # Grid in R
            xmin = Param.xmin
            xmax = Param.xmax
            nx = Param.nx
            dx = (xmax - xmin) / nx
            xvec = np.linspace(xmin, xmax, nx).reshape(-1, 1)
            pot_vec = self.V(xvec) 
            pot_filename = './data/pot.txt'
            np.savetxt(pot_filename, np.reshape(pot_vec, nx), header='%f %f %d\n' % (xmin,xmax,nx), comments="", fmt="%.10f")
            print("Potential is stored to: %s\n" % (pot_filename))

        if self.dim >= 2 :
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

            pot_vec = self.V(x2d)

            print ("Range of potential: [%.3f, %.3f]" % (min(pot_vec), max(pot_vec)) )

            pot_filename = './data/pot.txt'
            np.savetxt(pot_filename, np.reshape(pot_vec, (ny, nx)), header='%f %f %d\n%f %f %d' % (xmin,xmax,nx, ymin, ymax, ny), comments="", fmt="%.10f")

            print("Potential V is stored to: %s\n" % (pot_filename))

