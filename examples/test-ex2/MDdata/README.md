# Generate trajectory data of alanine dipeptide using NAMD

This directory contains files for generating training data for the alanine dipeptide example.

- [./abf-varying-20ns](./abf-varing-20ns): directory that contains configuration files of MD simulation (20ns) under adaptive biasing force (ABF).
- [./abf-fixed-100ns-0.7](./abf-fixed-100ns-0.7): directory that contains configuration files of MD simulation (100ns) under fixed biasing force.
- [rescale_abf_force.py](./rescale_abf_force.py): script that generates the (fixed) biasing force for the simulation of length 100ns. The force is computed by rescaling the biasing force estimated using ABF with rescaling factor 0.7.

### First, run ABF simulation of 20ns

1. enter the directory 
```
   cd abf-varying-20ns/
```
2. energy minimization 
```
   namd2 ./minvac.conf
```

	This step generates files *minvaco.vel* and *minvaco.coor*, which will provide initial veolcities and coordinates for the next step.

3. equilibration 

```
namd2 ./equilvac.conf
```

	The final coordinates and velocities are stored in the files *equilvaco.coor* and *equilvaco.vel*, respectively.

4. ABF simulation 
```
   namd2 ./colvars.conf
```

	This step performs a ABF simulation, using the collective variables defined in [colvars.in](./abf-varing-20ns/colvars.in).

	The information on the estimated biasing forces will be stored in the files *colvarso.count* and *colvarso.grad*.

### Second, run a 20ns simulation under fixed biasing force

1. prepare the rescaled biasing force
```
   python ./rescale_abf_force.py
```
2. copy the initial states generated in a previous step.
```
   cp ./abf-varying-20ns/equilvaco.* ./abf-fixed-100ns-0.7/
```
3. start the simulation 
```
   cd ./abf-fixed-100ns-0.7
   namd2 ./colvars.conf
```

