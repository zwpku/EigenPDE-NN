# EigenPDE-NN
An Eigenvalue PDE Solver by Neural Networks

This Python package trains artificial neural networks to solve the leading (in ascending order) eigenvalues/eigenfunctions of the eigenvalue PDE problem 
<img src="https://render.githubusercontent.com/render/math?math=-\mathcal{L}f=\lambda f">, where ![formula](https://render.githubusercontent.com/render/math?math=\mathcal{L}) is the infiniesimal generator of certain type of diffusion processes whose invariant measure is ![formula](https://render.githubusercontent.com/render/math?math=\mu).

A trajectory data (possibly biased with biasing weights) is needed as training data. Such data can be generated either 
- using a numerical scheme (e.g., Euler-Maruyama scheme), for diffusion processes whose stochastic differential equation (SDE) is given explicitly;  or 
- using a molecular simulation package (e.g., NAMD), for molecular systems.

#### References:
<a id="1"> [1] </a> Solving eigenvalue PDEs of metastable diffusion processes using artificial neural networks, W. Zhang, T. Li and Ch. Sch&uuml;tte, 2021, 
[https://arxiv.org/abs/2110.14523](https://arxiv.org/abs/2110.14523)

## Note
The code in this repository was used to produce the numerical results in the paper [[1]](#1). See instructions for the [two concrete examples](#two-examples) below.

Based on this repository, a Python package called [ColVars-Finder](https://github.com/zwpku/colvars-finder) with more detailed documentation was developed for finding collective variables of molecular systems.

## Preparation
#### 1. Dependances 

- [PyTorch](https://pytorch.org/)

- [MDAnalysis](https://www.mdanalysis.org/), used to load MD data. 

- [slepc4py](https://pypi.org/project/slepc4py/) and [petsc4py](https://pypi.org/project/petsc4py/), needed for solving 2D eigenvalue PDE problems using finite volume method. In some examples, the solution given by finite volume method can be used to compare with the neural network solution. 
These two packages are not needed if one just wants to solve a PDE problem by training neural networks.

#### 2. Download the code 

```
	git clone https://github.com/zwpku/EigenPDE-NN.git
```

## Two examples 
The directory [./examples](examples) contains necessary files of the numerical examples in the paper [[1]](#1).

### Example 1: A 50-dimensional system 

The dynamics of this system obeys the SDE <img src="https://render.githubusercontent.com/render/math?math=dX_t = -\nabla V(X_t)dt%2b\sqrt{2\beta^{-1}}dW_t"> where <img src="https://render.githubusercontent.com/render/math?math=V:\mathbb{R}^{50}\rightarrow\mathbb{R}"> is a potential function that has 3 metastable regions. The training data for this example is generated by directly sampling the system's SDE using Euler-Maruyama scheme.

#### Steps to solve the eigenvalue PDE:

1. Enter the directory corresponding to this example

```
    cd ./EigenPDE-NN
    cd ./examples/test-ex1-50d
```

2. Generate trajectory data

  Run the script [main.py](examples/test-ex1-50d/main.py) by `python ./main.py`, and choose task 0 from input.

3. Train neural networks

  Run the script [main.py](examples/test-ex1-50d/main.py) by `python ./main.py`, and choose task 3 from input.

### Example 2: Alanine Dipeptide example 

This example aims at solving the eigenvalue PDE of a simple molecular system in vacuum called *alanine dipeptide*.  The training data is generated using the molecular simulation package NAMD. To overcome the sampling difficulties due to the strong metastability of the system, adaptive biasing force (ABF) method is used where two dihedral angles of the system are chosen as collective variables.

#### Steps to solve the eigenvalue PDE:

1. Enter the directory corresponding to this example

```
    cd ./EigenPDE-NN
    cd ./examples/test-ex2
```

2. Generate MD data
  See the steps in [README](examples/test-ex2/MDdata/README.md).

3. Prepare data for training 
  Run the script [main.py](examples/test-ex2/main.py) by `python ./main.py`, and choose task 0 from input.

4. Train neural networks

  Run the script [main.py](examples/test-ex2/main.py) by `python ./main.py`, and choose task 3 from input.
	
