# EigenPDE-solver-NN
An Eigenvalue PDE Solver by Neural Networks

## Preparation
#### 1. Dependances 

- [MDAnalysis](https://www.mdanalysis.org/). It is used in order to load MD data. Not needed if one generates data using the script. 

- [slepc4py] and [petsc4py]. Both are used in solving 2D eigenvalue PDE problems using finite volume method. In some examples, the solution given by finite volume method can be used to compare with the solution obtained from training neural networks. Not needed if ones only wants to solve the problem by training neural networks.

#### 2. Download the code 

```
	git clone https://github.com/zwpku/EigenPDE-solver-NN.git
```

## Usage

#### 1. Enter the directory containing source codes and create a working directory.

```
  	cd ./Constrained-HMC
	cp -r working_dir working_dir_task1
	cd ./working_dir_task1
```

#### 2. View and modify the parameters in the configure file param.cfg 

#### 5. Choose a task by setting the value of task_id in ./main.py, and run the script 

```
   ./main.py
```

