#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:51:37 2020

@author: kissami
"""

from manapy.ddm import Domain, readmesh

from manapy.models.StreamerModel import StreamerModel, initialisation_streamer
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
filename = "rectangle_st.msh"
filename = os.path.join(MESH_DIR, filename)
    

dim = 2
readmesh(filename, dim=dim, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)

faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbcells = domain.nbcells

if RANK == 0: print("Start Computation")
cfl = 0.4
time = 0
tfinal = 5.25e-8
miter = 0
niter = 1
Pinit = 25000.
saving_at_node = 1
order = 2

boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }
values = {"in" : Pinit,
          "out": 0.,
          }


Model = StreamerModel(domain, terms=['convective', 'source', 'dissipation'], order=order)

Model.P = Variable(domain=domain, BC=boundaries, values=values)


ne = Model.ne
ni = Model.ni
u  = Model.u
v  = Model.v
w  = Model.w
Ex = Model.Ex
Ey = Model.Ey
Ez = Model.Ez
P  = Model.P

initialisation_streamer(ne.cell, ni.cell, u.cell, v.cell, Ex.cell, Ey.cell, P.cell, cells.center, Pinit=Pinit)

Model.initiate_model(solver="mumps")


if RANK == 0: print("Start loop")

#loop over time
while time < tfinal:
    
    Model.solve_linearSystem()
    
    #update time step
    Model.update_time_step(cfl=cfl)
    time = time + Model.time_step
    #saving 50 vtk file
    tot = int(tfinal/Model.time_step/50)+1
    
    #update solution   
    Model.update_solution()
    
    ne.interpolate_celltonode()
    
    #save vtk files for the solution
    if niter== 1 or niter%tot == 0:
        if saving_at_node:
            domain.save_on_node(Model.time_step, time, niter, miter, value=ne.node)
        else:
            domain.save_on_cell(Model.time_step, time, niter, miter, value=ne.cell)
        miter += 1

    niter += 1

if saving_at_node:
    domain.save_on_node(Model.time_step, time, niter, miter, value=ne.node)
else:
    domain.save_on_cell(Model.time_step, time, niter, miter, value=ne.cell)    

stop = timeit.default_timer()

Model.destroy()
