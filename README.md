# EigenPDE-solver-NN
An Eigenvalue PDE Solver by Neural Networks

## Preparation
#### 1. Dependances (optional)

- [MDAnalysis](https://www.mdanalysis.org/) is used to load MD data. It is **not** needed if one generates training data by sampling SDE (i.e. when parameter namd_data_flag=False [params.cfg](working_dir/params.cfg). 

- [slepc4py](https://pypi.org/project/slepc4py/) and [petsc4py](https://pypi.org/project/petsc4py/) are used in solving 2D eigenvalue PDE problems using finite volume method. In some examples, the solution given by finite volume method can be used to compare with the solution obtained from training neural networks. 
These two packages are **not** needed if one only wants to solve the problem by training neural networks.

- [torch.distributed](https://pytorch.org/docs/stable/distributed.html) is used to enable parallel training of neural networks. The current implementation uses MPI as backend, and this requires that PyTorch is built from source. This package is **not** needed for sequential training (recommended).

#### 2. Download the code 

```
	git clone https://github.com/zwpku/EigenPDE-solver-NN.git
```

## Usage

#### 1. Enter the directory containing source codes and create a clean working directory.

```
  	cd ./EigenPDE-solver-NN
	cp -r working_dir working_dir_task1
	cd ./working_dir_task1
```

#### 2. View and modify the parameters in the configure file [params.cfg](working_dir/params.cfg).
   Set num_processor = 1 for standard training.  

#### 3. Choose a task (see the scripts under directory [utils](utils)) by setting the value of task_id in [main.py](working_dir/main.py), and run the script 

```
   ./main.py
```


