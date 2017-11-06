# siesta_utils

This library contains utility functions that can be used to analyze the output of the 
[SIESTA](https://departments.icmab.es/leem/siesta/) electronic structure code.

## Installation

Change into the root directory of the repository and type 
```
pip install -e .
```
Dependencies:

numpy 1.13
pandas 0.20
matplotlib 2.0

## grid.py 

This module provides functions to analyze observables defined on a three dimensional grid, such as the charge distribution

## mat.py

Functions to import and manipulate matrices that are saved in the sparse density format used by SIESTA. These include
density, overlap and hamiltonian matrix.
