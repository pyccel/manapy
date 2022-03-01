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

from manapy.models.StreamerModel.tools import (initialisation_streamer, compute_velocity, 
                                               compute_el_field, update_rhs_glob,
                                               update_streamer_flux, time_step_streamer,
                                               explicitscheme_source_streamer,
                                               explicitscheme_dissipative_streamer)
                                               
from manapy.fvm.pyccel_fvm import explicitscheme_convective_2d

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
 
filename = "rectangle_st.msh"

#File name
filename = os.path.join(MESH_DIR, filename)
dim = 2
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
cfl = 0.4
time = 0
tfinal = 5.25e-8
miter = 0
niter = 1
Pinit = 25000.
saving_at_node = 1
order = 2

# import sys; sys.exit()
boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }
values = {"in" : Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
ni = Variable(domain=domain)
u  = Variable(domain=domain)
v  = Variable(domain=domain)
w  = Variable(domain=domain)
Ex = Variable(domain=domain)
Ey = Variable(domain=domain)
Ez = Variable(domain=domain)
P  = Variable(domain=domain, BC=boundaries, values=values)

####Initialisation
initialisation_streamer(ne.cell, ni.cell, u.cell, v.cell, Ex.cell, Ey.cell, P.cell, cells.center, Pinit=Pinit)

f = lambda x, y, z : Pinit * (1. - x)
L = LinearSystem(domain=domain, var=P, solver="mumps", debug=None)
L.assembly()


rez_ne = np.zeros(nbcells)
rez_ni = np.zeros(nbcells)
dissip_ne = np.zeros(nbcells)
dissip_ni = np.zeros(nbcells)
src_ne = np.zeros(nbcells)
src_ni = np.zeros(nbcells)

# rhs_updated = np.zeros(L.localsize)
rhs_updated = np.zeros(L.globalsize)

#loop over time
while time < tfinal:
    
    #update rhs
    update_rhs_glob(ne.cell, ni.cell, cells.loctoglob, rhs_updated) 
    rhs = COMM.reduce(rhs_updated, op=MPI.SUM, root=0)
    
    L.solve(compute_grad=True, rhs=rhs)
    
    compute_el_field(P.gradfacex, P.gradfacey, P.gradfacez, Ex.face, Ey.face, Ez.face)
    compute_velocity(Ex.face, Ey.face, Ez.face, u.face, v.face, w.face, Ex.cell, Ey.cell, Ez.cell, u.cell, v.cell, w.cell, 
                     cells.faceid, dim)
    
    #####calculation of the time step
    dt = time_step_streamer(u.cell, v.cell, w.cell, Ex.cell, Ey.cell, Ez.cell, cfl, faces.normal, faces.mesure, 
                            cells.volume, cells.faceid, dim)
    d_tt = COMM.allreduce(dt, MPI.MIN)
    tot = int(tfinal/d_tt/50)+1
    
    time = time + d_tt

    explicitscheme_source_streamer(ne.cell, u.cell, v.cell, w.cell, Ex.cell, Ey.cell, Ez.cell, src_ne, src_ni,
                                   cells.center, br=0)
    
    ne.compute_cell_gradient()
    explicitscheme_convective_2d(rez_ne, ne.cell, ne.ghost, ne.halo, u.face, v.face, w.face,
                                 ne.gradcellx, ne.gradcelly, ne.gradcellz, ne.gradhalocellx, 
                                 ne.gradhalocelly, ne.gradhalocellz, ne.psi, ne.psihalo, 
                                 cells.center, faces.center, halos.centvol, faces.ghostcenter,
                                 faces.cellid, faces.mesure, faces.normal, faces.halofid, faces.name, 
                                 domain.innerfaces, domain.halofaces, domain.boundaryfaces, 
                                 domain.periodicboundaryfaces, cells.shift, order=order)
    
    ne.compute_face_gradient()
    explicitscheme_dissipative_streamer(u.face, v.face, w.face, Ex.face, Ey.face, Ez.face, ne.gradfacex, ne.gradfacey, ne.gradfacez, 
                                        faces.cellid, faces.normal, faces.name, dissip_ne)
    
    update_streamer_flux(ne.cell, ni.cell, rez_ne, rez_ni, dissip_ne, dissip_ni, src_ne, src_ni, d_tt, cells.volume)

    #save vtk files for the solution
    ne.interpolate_celltonode()
    
    if niter== 1 or niter%tot == 0:
        if saving_at_node:
            
            domain.save_on_node(d_tt, time, niter, miter, value=ne.node)
        else:
            domain.save_on_cell(d_tt, time, niter, miter, value=P.cell)
        miter += 1

    niter += 1

if saving_at_node:
    domain.save_on_node(d_tt, time, niter, miter, value=ne.node)
else:
    domain.save_on_cell(d_tt, time, niter, miter, value=P.cell)   

L.destroy()
