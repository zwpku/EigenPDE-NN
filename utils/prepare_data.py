#!/usr/bin/env python3

# Load MD data and save it to txt file

import sys

# Load directories containing source codes
sys.path.append('../src/')

import read_parameters 
import training_data

Param = read_parameters.Param()
data_proc = training_data.PrepareData(Param) 
data_proc.prepare_data()
