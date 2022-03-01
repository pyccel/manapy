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
import os

from mpi4py import MPI
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
 
filename = "carre.msh"
filename = os.path.join(MESH_DIR, filename)

def test2d_1():
        
    dim = 2
    readmesh(filename, dim=dim, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim)
    
    cells = domain.cells
    nbcells = domain.nbcells
    
    boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }
    values = {"in" : 25000.,
              "out": 0.
              }
    
    w = Variable(domain=domain, BC=boundaries, values=values)
    
    # import sys; sys.exit()
    L = LinearSystem(domain=domain, var=w, solver="mumps")
    L.assembly()
    L.solve()
    L.destroy()
    
    # interpolate value on node
    w.interpolate_celltonode()
    
    # save value on node using paraview
    domain.save_on_node(0,0,0,0, value=w.node)
    
    f = lambda x, y, z : 25000. * (1. - x)
    fexact = np.zeros(nbcells)
    
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1], 0.)
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)

def test2d_2():
    
    dim = 2
    readmesh(filename, dim=dim, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim)
    
    cells = domain.cells
    nbcells = domain.nbcells
    
    f = lambda x, y, z : x**2 + y**2
      
    boundaries = {"in" : "dirichlet",
                  "out" : "dirichlet",
                   "upper":"dirichlet",
                   "bottom":"dirichlet"
                  }
    values = {"in" : f,
              "out": f,
               "upper":f,
               "bottom":f
              }
    w = Variable(domain=domain, BC=boundaries, values=values)
    L = LinearSystem(domain=domain, var=w, solver="mumps")

    if RANK == 0:
        rhs = np.zeros(L.globalsize)
        rhs[:] = 4.
    else:
        rhs = None
        
    L.assembly()
    L.solve(rhs=rhs)
    L.destroy()
    
    # interpolate value on node
    w.interpolate_celltonode()
    
    #save value on node using paraview
    domain.save_on_node(0,0,0,1,value=w.node) 
    
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1], 0.)
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)
    
def test2d_3():
    
    dim = 2
    readmesh(filename, dim=dim, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim)
    
    cells = domain.cells
    nbcells = domain.nbcells
    
    f = lambda x, y, z : x*(x-1) + y*(y-1)
      
    boundaries = {"in" : "dirichlet",
                  "out" : "dirichlet",
                  "upper":"dirichlet",
                  "bottom":"dirichlet"
                  }
    values = {"in" : f,
              "out": f,
              "upper":f,
              "bottom":f
              }
    
    w = Variable(domain=domain, BC=boundaries, values=values)
    
    L = LinearSystem(domain=domain, var=w, solver="mumps")
    
    if RANK == 0:
        rhs = np.zeros(L.globalsize)
        rhs[:] = 4.
    else:
        rhs = None
        
    L.assembly()
    L.solve(rhs=rhs)
    L.destroy()
    
    # interpolate value on node
    w.interpolate_celltonode()
    
    #save value on node using paraview
    domain.save_on_node(0,0,0,2,value=w.node) 
    
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1], 0.)
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)
    
def test2d_4():
    
    dim = 2
    readmesh(filename, dim=dim, periodic=[0,0,0])
    
    #Create the informations about cells, faces and nodes
    domain = Domain(dim=dim)
    
    cells = domain.cells
    nbcells = domain.nbcells
    
    alpha = 4
    f = lambda x, y, z : np.sin(alpha*np.pi*x - np.pi/3) * np.sin(alpha*np.pi*y - np.pi/3)
      
    boundaries = {"in" : "dirichlet",
                  "out" : "dirichlet",
                  "upper":"dirichlet",
                  "bottom":"dirichlet"
                  }
    values = {"in" : f,
              "out": f,
              "upper":f,
              "bottom":f
              }
    w = Variable(domain=domain, BC=boundaries, values=values)
    
   
    L = LinearSystem(domain=domain, var=w, solver="mumps")
    rhs = np.zeros(L.globalsize)
    for i in range(nbcells):
        x = cells.center[i][0];  y = cells.center[i][1]
        rhs[cells.loctoglob[i]] = -2*(alpha*np.pi)**2*(np.sin(alpha*np.pi*x - np.pi/3) * np.sin(alpha*np.pi*y - np.pi/3))
    
    rhs = COMM.reduce(rhs, op=MPI.SUM, root=0)
    
    L.assembly()
    L.solve(rhs=rhs)
    L.destroy()
    
    # interpolate value on node
    w.interpolate_celltonode()
    
    #save value on node using paraview
    domain.save_on_node(0,0,0,4, value=w.node) 
    
    fexact = np.zeros(nbcells)
    for i in range(nbcells):
        fexact[i] = f(cells.center[i][0], cells.center[i][1], 0.)
    errorl2 = w.norml2(fexact, order=1)  
    #compute error
    print("l2 norm is ", errorl2)
       
# test 1
test2d_1()

# test 2
test2d_2()

# test 3
test2d_3()

# test 4
test2d_4()

