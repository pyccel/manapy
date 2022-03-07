#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:06:35 2022

@author: kissami
"""

from mpi4py import MPI
import os

import numpy as np

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from numba import njit

from manapy.ddm import readmesh
from manapy.ddm import Domain                                       
 
from manapy.ast import Variable

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
filename = "cube.msh"
filename = os.path.join(MESH_DIR, filename)

dim = 3
readmesh(filename, dim=dim)

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)

faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

w = Variable(domain=domain)
w.cell[:] = 10.
w.interpolate_celltonode()

domain.save_on_node(value=w.node)
