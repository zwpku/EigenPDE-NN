# Generate trajectory data of alanine dipeptide using NAMD

This directory contains files for generating training data for the alanine dipeptide example.

- [./abf-varying-20ns](./abf-varing-20ns): directory that contains configuration files of MD simulation (20ns) under adaptive biasing force (ABF).
- [./abf-fixed-100ns-0.7](./abf-fixed-100ns-0.7): directory that contains configuration files of MD simulation (100ns) under fixed biasing force.
- [rescale_abf_force.py](./rescale_abf_force.py): script that generates files of the (fixed) biasing force for the simulation of length 100ns. The force is computed by rescaling the biasing force estimated using ABF with rescaling factor 0.7.

### ABF simulation of 20ns
1. enter the directory 
```
cd abf-varying-20ns/
```
2. energy minimization 
```
namd2 ./minvac.conf
```
This step generates the files *minvac.vel* and *minvac.coor*

3. equilibration 

```
namd2 ./equilvac.conf
```
This step generates the file *equilvaco.coor*

4. ABF simulation 
```
namd2 ./colvars.conf
```
The step saves the information of the biasing forces in the files *colvarso.count* and *colvarso.grad*.

### Simulation of 20ns under fixed biasing force 
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

