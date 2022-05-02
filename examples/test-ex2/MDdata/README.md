# Generate trajectory data of alanine dipeptide using NAMD

This directory contains files for generating training data for the alanine dipeptide example.

- [./abf-varying-20ns](./abf-varing-20ns): directory that contains configuration files of MD simulation (20ns) under adaptive biasing force (ABF).
- [./abf-fixed-100ns-0.7](./abf-fixed-100ns-0.7): directory that contains configuration files of MD simulation (100ns) under fixed biasing force.
- [rescale_abf_force.py](./rescale_abf_force.py): script that generates the (fixed) biasing force for the simulation of length 100ns. 

### First, run ABF simulation of 20ns

1. Enter the directory:  `cd abf-varying-20ns/`

2. Energy minimization:  `namd2 ./minvac.conf`

	This step generates files *minvaco.vel* and *minvaco.coor*, which will provide initial veolcities and coordinates for the next step.

3. Equilibration: `namd2 ./equilvac.conf`

	The final coordinates and velocities are stored in the files *equilvaco.coor* and *equilvaco.vel*, respectively.

4. ABF simulation: `namd2 ./colvars.conf`

	This step performs a ABF simulation, using the collective variables defined in [colvars.in](./abf-varing-20ns/MDdata/colvars.in). The information on the estimated biasing forces will be stored in the files *colvarso.count* and *colvarso.grad*.

### Second, run a 20ns simulation under fixed biasing force

1. Change to [MDdata](.), and Prepare the rescaled biasing force

   ```
   python ./rescale_abf_force.py
   ```

   The force is computed by rescaling the biasing force estimated using ABF above with rescaling factor 0.7. Files for the rescaled force will be generated under [./abf-fixed-100ns-0.7](./abf-fixed-100ns-0.7).

2. Prepare the initial coordinates/velocities (generated in a previous step)

   ```
   cp ./abf-varying-20ns/equilvaco.* ./abf-fixed-100ns-0.7/
   ```

3. Start the simulation 

   ```
   cd ./abf-fixed-100ns-0.7
   namd2 ./colvars.conf
   ```

   This step performs a simulation of 100ns, using the rescaled biasing force prepared above.

