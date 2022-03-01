#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:51:37 2020

@author: kissami
"""

from manapy.ddm import Domain, readmesh

from manapy.models.SWModel import ShallowWaterModel, initialisation_SW

from manapy.ast import Variable

import timeit
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
    BASE_DIR = os.path.join(BASE_DIR , '..', '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')

#File name
filename = "rectangle_sw.msh"
filename = os.path.join(MESH_DIR, filename)
    

dim = 2
readmesh(filename, dim=dim, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)

faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

if RANK == 0: print("Start Computation")

#TODO tfinal
cfl = 0.7
time = 0
tfinal = .1
order = 2
saving_at_node = 1

parameters = {}
parameters["Manning"] = 0.
parameters["CoriolisForce"] = 0.
parameters["Dx"] = 0.
parameters["Dy"] = 0.


SWModel = ShallowWaterModel(domain, terms=['source', 'dissipation', 'coriolis', 'friction'], 
                            parameters=parameters, order=order)

h  = SWModel.h
hu = SWModel.hu
hv = SWModel.hv
hc = SWModel.hc
Z  = SWModel.Z

boundaries = {"in" : "neumann",
              "out" : "neumann",
              "upper":"nonslip",
              "bottom":"nonslip"
              }
values = {}

SWModel.hu = Variable(domain=domain, BC=boundaries, terms=['convective', 'source', 'dissipation', 'coriolis', 'friction'], values=values)
SWModel.hv = Variable(domain=domain, BC=boundaries, terms=['convective', 'source', 'dissipation', 'coriolis', 'friction'], values=values)

####Initialisation
choix = -1
# import sys;
initialisation_SW(h.cell, hu.cell, hv.cell, hc.cell, Z.cell, cells.center)

#update time step
SWModel.update_time_step(cfl=cfl)

#saving 50 vtk file
tot = int((tfinal-time)/SWModel.time_step/50)
miter = 0
niter = 1

if RANK == 0:
    print("Start loop")
    
start = timeit.default_timer()

#loop over time
while time < tfinal:
    
    time = time + SWModel.time_step
    
    #update the ghost values for the boundary conditions
    SWModel.update_values()
    
    #update solution   
    SWModel.update_solution()
    
    #update time step
    SWModel.update_time_step(cfl=cfl)
    
    h.interpolate_celltonode()
    
    #save vtk files for the solution
    if niter== 1 or niter%tot == 0:
        if saving_at_node:
            domain.save_on_node(SWModel.time_step, time, niter, miter, value=h.node)
        else:
            domain.save_on_cell(SWModel.time_step, time, niter, miter, value=h.cell)
        miter += 1

    niter += 1

if saving_at_node:
    domain.save_on_node(SWModel.time_step, time, niter, miter, value=h.node)
else:
    domain.save_on_cell(SWModel.time_step, time, niter, miter, value=h.cell)    

stop = timeit.default_timer()

if RANK == 0: print(stop - start)
