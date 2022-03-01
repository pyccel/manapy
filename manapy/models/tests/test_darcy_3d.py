#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:13:21 2022

@author: kissami
"""

from mpi4py import MPI
import timeit

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from manapy.ddm import readmesh
from manapy.ddm import Domain

from manapy.tools.pyccel_tools import initialisation_gaussian_2d, update_new_value, time_step
                          
from manapy.fvm.pyccel_fvm import (explicitscheme_convective_3d, explicitscheme_dissipative)

from manapy.ast import Variable, LinearSystem

import numpy as np
import os

start = timeit.default_timer()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
filename = "cube.msh"

#File name
filename = os.path.join(MESH_DIR, filename)
dim = 3
readmesh(filename, dim=dim, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)

faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

end = timeit.default_timer()
if RANK == 0:
    print("Time to create the domain", end -start)


#TODO tfinal
if RANK == 0: print("Start Computation")
cfl = 0.8
time = 0
tfinal = .05
miter = 0
niter = 1
Pinit = 10.
saving_at_node = 1
order = 2

Dxx=0.1
Dyy=0.
Dzz=0.

# import sys; sys.exit()
boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann",
              "front":"neumann",
              "back":"neumann"
              }
values = {"in" : Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
u  = Variable(domain=domain)
v  = Variable(domain=domain)
w  = Variable(domain=domain)
P  = Variable(domain=domain, BC=boundaries, values=values)

####Initialisation
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, w.cell, P.cell, cells.center, Pinit=Pinit)#, 


f = lambda x, y, z : Pinit * (1. - x)
L = LinearSystem(domain=domain, var=P, solver="mumps")
L.assembly()


rhs_updated = np.zeros(nbcells)
rez_ne = np.zeros(nbcells)
dissip_ne = np.zeros(nbcells)
src_ne = np.zeros(nbcells)
src_ni = np.zeros(nbcells)

#loop over time
while time < tfinal:
    
    L.solve(compute_grad=True)
    
    #TODO -1
    u.face[:] = P.gradfacex[:]
    v.face[:] = P.gradfacey[:]
    w.face[:] = P.gradfacez[:]
    
    u.interpolate_facetocell()
    v.interpolate_facetocell()
    w.interpolate_facetocell()
    
    ne.update_values()
    
    ######calculation of the time step
    dt_c = time_step(u.cell, v.cell, w.cell, cfl, faces.normal, faces.mesure, cells.volume, cells.faceid,
                     dim, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz)

    d_t = COMM.allreduce(dt_c, MPI.MIN)
    tot = int(tfinal/d_t/50)+1

    time = time + d_t
    
    ne.compute_cell_gradient()
    explicitscheme_convective_3d(rez_ne, ne.cell, ne.ghost, ne.halo, u.face, v.face, w.face,
                                 ne.gradcellx, ne.gradcelly, ne.gradcellz, ne.gradhalocellx, 
                                 ne.gradhalocelly, ne.gradhalocellz, ne.psi, ne.psihalo, 
                                 cells.center, faces.center, halos.centvol, faces.ghostcenter,
                                 faces.cellid, faces.mesure, faces.normal, faces.halofid, faces.name, 
                                 domain.innerfaces, domain.halofaces, domain.boundaryfaces, 
                                 domain.periodicboundaryfaces, cells.shift, order=order)
    
    ne.compute_face_gradient()
    explicitscheme_dissipative(ne.gradfacex, ne.gradfacey, ne.gradfacez, faces.cellid, faces.normal, faces.name, dissip_ne,
                               Dxx=Dxx, Dyy=Dyy, Dzz=Dzz)

    # import sys; sys.exit()
    update_new_value(ne.cell, u.cell, v.cell, P.cell, rez_ne,  dissip_ne, src_ne, d_t, cells.volume)    
    
    #save vtk files for the solution
    ne.interpolate_celltonode()
    
    if niter== 1 or niter%tot == 0:
        if saving_at_node:
            
            domain.save_on_node(d_t, time, niter, miter, value=ne.node)
        else:
            domain.save_on_cell(d_t, time, niter, miter, value=ne.cell)
        miter += 1

    niter += 1

L.destroy()
