#!/usr/bin/env python3

# Load MD data and save it to txt file

import sys

import read_parameters 
from data_generator import PrepareData

training_data = True

if len(sys.argv) > 1 and sys.argv[1] == 'test' :
    training_data = False

Param = read_parameters.Param(use_sections={'grid'})
data_proc = PrepareData(Param) 
data_proc.prepare_data(training_data)
