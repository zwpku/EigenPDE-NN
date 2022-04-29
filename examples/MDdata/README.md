# EigenPDE-solver-NN

namd2 ./minvac.conf
cd abf-varying-20ns/
minvac_0.vel
minvac_0.coor

namd2 ./equilvac.conf
equilvaco.coor
namd2 ./colvars.conf
colvarso.count
colvarso.grad
python ./rescale_abf_force.py
cp ../abf-varying-20ns/equilvaco.* .
states_100ns_7e-1-noalign.txt
