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

5. Evaluate the trained neural networks on training (test) data

	  Run the script [main.py](./main.py) by `python ./main.py`, and choose task 5 from input.

	  Note: This step evaluates each trained eigenfunction on training (test) data. This step generates text files that are needed to plot eigenfunctions on scattered data in the space of two dihedral angles.

#### Generate plots:
1. Data

	Run the script [main.py](./main.py) by `python ./main.py`, and choose task 2 from input.

	This will generate plots of histogram and counts of weights along trajectory data.

2. Training results

	Use scripts under [plot_scripts](../../plot_scripts/) **as templates** to generate plots. Before use, set the value of `task_id` in [common.py](../../plot_scripts/common.py).  The plots will be created under `./fig/`.

  - [plot_log_info.py](../../plot_scripts/plot_log_info.py): plot losses.

  - [scatter_eigenvecs_angle_space.py](../../plot_scripts/scatter_eigenvecs_angle_space.py): display eigenfunctions on training (test) data in the space of two dihedral angles.


