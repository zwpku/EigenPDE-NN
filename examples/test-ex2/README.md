## Example 2: Alanine Dipeptide example 

This example aims at solving the eigenvalue PDE of a simple molecular system in vacuum called *alanine dipeptide*.  The training data is generated using the molecular simulation package NAMD. To overcome the sampling difficulties due to the strong metastability of the system, adaptive biasing force (ABF) method is used where two dihedral angles of the system are chosen as collective variables.

#### Steps to solve the eigenvalue PDE:

1. Enter the directory corresponding to this example

```
    cd ./EigenPDE-NN
    cd ./examples/test-ex2
```

2. Generate MD data

	Enter [./MDdata](./MDdata), and follow the steps in [README](./MDdata/README.md).

3. Prepare data for training 

	  Run the script [main.py](./main.py) by `python ./main.py`, and choose task 0 from input.

	  Note: This step generates several output files under `./data` by reading the DCD trajectory file from the previous step, as well as the PDB and PSF files. One file, whose name is specified by the value of `data_filename_prefix` in [params.cfg](./params.cfg), contains states and weights of the selected atoms that are used in training.

4. Train neural networks

	  Run the script [main.py](./main.py) by `python ./main.py`, and choose task 3 from input.

	  This will learn the first eigenvalue/eigenfunction.

