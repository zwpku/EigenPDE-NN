#!/usr/bin/env python3

import read_parameters 
import namd_loader_dipeptide

Param = read_parameters.Param()
namd_loader = namd_loader_dipeptide.namd_data_loader(Param) 
namd_loader.plot_namd_data()

