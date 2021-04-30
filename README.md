# EigenPDE-solver-NN
An Eigenvalue PDE Solver by Neural Networks

## Preparation
#### 1. Dependances 

- [MDAnalysis](https://www.mdanalysis.org/) is used in order to load MD data. Not needed if one generates training data using the script [./utils/generate_sample_data.py](./utils/generate_sample_data.py). 

- [slepc4py](https://pypi.org/project/slepc4py/) and [petsc4py](https://pypi.org/project/petsc4py/) are used in solving 2D eigenvalue PDE problems using finite volume method. In some examples, the solution given by finite volume method can be used to compare with the solution obtained from training neural networks. Not needed if ones only wants to solve the problem by training neural networks.

#### 2. Download the code 

```
	git clone https://github.com/zwpku/EigenPDE-solver-NN.git
```

## Usage

#### 1. Enter the directory containing source codes and create a working directory.

```
  	cd ./EigenPDE-solver-NN
	cp -r working_dir working_dir_task1
	cd ./working_dir_task1
```

#### 2. View and modify the parameters in the configure file [param.cfg](./working_dir/param.cfg)

#### 5. Choose a task by setting the value of task_id in [./main.py](./working_dir/main.py), and run the script 

```
   ./main.py
```


