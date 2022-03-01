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


@njit
def node_for_interpolation_3d(w_cell, w_node, w_ghost, w_halo, nodefid, cellfid, halofid, 
                              cellcenter, vertexcenter, ghostcenter, halocenter, name):
    '''This function compute the value xcenter ycenter yenter for the 5 points arround all faces'''
    
    ValForInterp = np.zeros((len(nodefid), 5))
    xCenterForInterp = np.zeros((len(nodefid), 5))
    yCenterForInterp = np.zeros((len(nodefid), 5))
    zCenterForInterp = np.zeros((len(nodefid), 5))

    nbfaces = len(nodefid)
    for i in range(nbfaces):
        
        ValForInterp[i][0:3] = w_node[nodefid[i][:]]
        ValForInterp[i][3]   = w_cell[cellfid[i][0]]
        
        xCenterForInterp[i][0:3] = vertexcenter[nodefid[i][:], 0]
        xCenterForInterp[i][3] = cellcenter[cellfid[i][0]][0]
        
        yCenterForInterp[i][0:3] = vertexcenter[nodefid[i][:], 1]
        yCenterForInterp[i][3] = cellcenter[cellfid[i][0]][1]
        
        zCenterForInterp[i][0:3] = vertexcenter[nodefid[i][:], 2]
        zCenterForInterp[i][3] = cellcenter[cellfid[i][0]][2]
        
        
        if name[i] == 0:
            ValForInterp[i][4] = w_cell[cellfid[i][1]]
            xCenterForInterp[i][4] = cellcenter[cellfid[i][1]][0]
            yCenterForInterp[i][4] = cellcenter[cellfid[i][1]][1]
            zCenterForInterp[i][4] = cellcenter[cellfid[i][1]][2]
            
        elif name[i] == 10: 
            ValForInterp[i][4] = w_halo[halofid[i]]
            xCenterForInterp[i][4] = halocenter[halofid[i]][0]
            yCenterForInterp[i][4] = halocenter[halofid[i]][1]
            zCenterForInterp[i][4] = halocenter[halofid[i]][2]
            
        else:
            ValForInterp[i][4] = w_ghost[i]
            xCenterForInterp[i][4] = ghostcenter[i][0]
            yCenterForInterp[i][4] = ghostcenter[i][1]
            zCenterForInterp[i][4] = ghostcenter[i][2]
            
    return ValForInterp, xCenterForInterp, yCenterForInterp, zCenterForInterp




ValForInterp, xCenterForInterp, yCenterForInterp, zCenterForInterp = node_for_interpolation_3d(w.cell, w.node, w.ghost, w.halo, 
                                                                                               faces.nodeid, faces.cellid, faces.halofid, 
                                                                                               cells.center, nodes.vertex, faces.ghostcenter, 
                                                                                               halos.centvol, faces.name)