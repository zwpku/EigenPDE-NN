# EigenPDE-NN
An Eigenvalue PDE Solver by Neural Networks

This Python package trains artificial neural networks to solve the leading (in ascending order) eigenvalues/eigenfunctions of the eigenvalue PDE problem 
<img src="https://render.githubusercontent.com/render/math?math=-\mathcal{L}f=\lambda f">, where ![formula](https://render.githubusercontent.com/render/math?math=\mathcal{L}) is the infiniesimal generator of certain type of diffusion processes whose invariant measure is ![formula](https://render.githubusercontent.com/render/math?math=\mu).

A trajectory data (possibly biased with biasing weights) is needed as training data. Such data can be generated either 
- using a numerical scheme (e.g., Euler-Maruyama scheme), for diffusion processes whose stochastic differential equation (SDE) is given explicitly;  or 
- using a molecular simulation package (e.g., NAMD), for molecular systems.


#### Note:

The code in this repository was used to produce the numerical results in the paper [[1]](#1). See instructions for the [two concrete examples](#two-examples) below. Based on this repository, a Python package called [ColVars-Finder](https://github.com/zwpku/colvars-finder) with more detailed documentation was developed for finding collective variables of molecular systems.

#### References:
<a id="1"> [1] </a> Solving eigenvalue PDEs of metastable diffusion processes using artificial neural networks, W. Zhang, T. Li and Ch. Sch&uuml;tte, 2021, 
[https://arxiv.org/abs/2110.14523](https://arxiv.org/abs/2110.14523)

## Preparation
#### 1. Dependances 

- [PyTorch](https://pytorch.org/)

- [MDAnalysis](https://www.mdanalysis.org/), used to load MD data. 

- [NAMD](https://www.ks.uiuc.edu/Research/namd/), molecular simulation code. It is needed to run the second example below.

- [slepc4py](https://pypi.org/project/slepc4py/) and [petsc4py](https://pypi.org/project/petsc4py/), needed for solving 2D eigenvalue PDE problems using finite volume method. In some examples, the solution given by finite volume method can be used to compare with the neural network solution. 
These two packages are *not* needed if one just wants to solve a PDE problem by training neural networks.

#### 2. Download the code 

```
	git clone https://github.com/zwpku/EigenPDE-NN.git
```

## Two examples 
The directory [./examples](./examples) contains necessary files of two numerical examples studied in the paper [[1]](#1).
The first example is A 50-dimensional system, while the second example is the simple MD system *alanine dipeptide*.
Please refer to the README files in [./examples/test-ex1-50d](./examples/test-ex1-50d) and [./examples/test-ex2](./examples/test-ex2) for detailed instructions to run the simulations.

