#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:47:53 2022

@author: kissami
"""
from manapy.ddm import readmesh
from manapy.ddm import Domain

from manapy.ast import Variable, LinearSystem

import numpy as np
from mpi4py import MPI
import os

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
filename = "cube.msh"
#File name
filename = os.path.join(MESH_DIR, filename)


def test3d_1():
    dim = 3
    readmesh(filename, dim=dim, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim)
    cells = domain.cells
    nbcells = domain.nbcells
    
    f = lambda x, y, z : 25000. * (1. - x)
       
    boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann",
              "front":"neumann",
              "back":"neumann"              
              }
    values = {
            "in" : 25000.,
              "out": 0.,
              }
    w = Variable(domain=domain, BC=boundaries, values=values)
    L = LinearSystem(domain=domain, var=w, solver="spsolve")
    L.assembly()
    L.solve()
    L.destroy()
    
    # interpolate value on node
    w.interpolate_celltonode()
    
    #save value on node using paraview
    domain.save_on_node(0,0,0,0, value=w.node)
    
       
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1],  cells.center[i][2])
    
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)
   

def test3d_2():
    dim = 3
    readmesh(filename, dim=dim, periodic=[0,0,0])
    # Create the informations about cells, faces and nodes
    domain = Domain(dim=dim)
    cells = domain.cells
    nbcells = domain.nbcells
   
    f = lambda x, y, z : x**2 + y**2 + z**2
    
    boundaries = {"in" : "dirichlet",
                  "out" : "dirichlet",
                  "upper":"dirichlet",
                  "bottom":"dirichlet",
                  "front":"dirichlet",
                  "back":"dirichlet"
                  }
    values = {"in" : f,
              "out": f,
              "upper":f,
              "bottom":f,
              "front":f,
              "back":f
              }
    w = Variable(domain=domain, BC=boundaries, values=values)
    
    L = LinearSystem(domain=domain, var=w, solver="mumps")
   
    if RANK == 0:
        rhs = np.zeros(L.globalsize)
        rhs[:] = 6.
    else:
        rhs = None
        
    L.assembly()
    L.solve(rhs=rhs)
    L.destroy()
    
    # interpolate value on node
    w.interpolate_celltonode()
    
    # save value on node using paraview
    domain.save_on_node(0,0,0,1, value=w.node)
    
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1],  cells.center[i][2])
    
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)

# test 1
test3d_1()

# test 2
test3d_2()


