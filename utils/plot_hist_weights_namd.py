#!/usr/bin/env python3

import read_parameters 
import md_loader_dipeptide

Param = read_parameters.Param(use_sections={'training'})
md_loader = md_loader_dipeptide.md_data_loader(Param) 
md_loader.plot_md_data()

