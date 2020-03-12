#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:51:37 2020

@author: kissami
"""

from manapy import meshpart
from mpi4py import MPI
import numpy as np
import timeit
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0 :
    #reading gmsh file and partitioning into size subdomains
    meshpart.MeshPart(size, "rect_005.msh")
    #removing existing vtk files
    mypath = "results"
    if not os.path.exists(mypath):
        os.mkdir(mypath)
    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
comm.Barrier()

start = timeit.default_timer()

#generating local grid for each subdomain
g = {}
g  = meshpart.generate_mesh()


faces = g["faces"]
cells = g["cells"]
halos = g["halos"]
nodes = g["nodes"]

nbelements = len(cells.center)
nbfaces    = len(faces.name)
nbnodes    = len(nodes.vertex)

variables = tuple(['h', 'hu', 'hv'])
mystruct = np.dtype([('h', np.float64),
                       ('hu', np.float64),
                       ('hv', np.float64)])

w = np.recarray(nbelements, dtype = mystruct )
w_ghost = np.recarray(nbfaces, dtype = mystruct)


#compute the arrays needed for the mpi communication
if size > 1 :
    scount, sdepl, rcount, rdepl, taille, indSend, indRecv =  meshpart.prepare_comm(cells,
                                                              faces, nodes, halos)
    w_halosend = np.zeros(len(halos.halosInt), dtype = mystruct )
    w_halo = np.zeros(nbfaces, dtype = mystruct )
else:
    scount = sdepl = rcount = rdepl = taille = 0
    w_halo  = np.zeros(0, dtype = mystruct )

wn = w
###Initialisation
w = meshpart.initialisation(w, cells.center)
#


cfl = 0.5
####calculation of the time step
dt = meshpart.time_step(w, cfl, faces.normal, cells.volume, cells.faceid)
dt_i = np.zeros(1)
comm.Allreduce(dt, dt_i, MPI.MIN)
dt = np.float64(dt_i)
#
#
t = 0
Tfinal = 1

#saving 25 vtk file
T = int(Tfinal/dt/25)
m = 0
n = 0
#loop over time
while(t<Tfinal):

    t = t + dt

    ####calculation of the time step
    dt = meshpart.time_step(w, cfl, faces.normal, cells.volume, cells.faceid)
    dt_i = np.zeros(1)
    comm.Allreduce(dt, dt_i, MPI.MIN)
    dt = np.float64(dt_i)
    #saving 25 vtk file
    T = int(Tfinal/dt/25)

#    dt = 0.0008
    #update the ghost values for the boundary conditions
    w_ghost = meshpart.ghost_value(w, w_ghost, faces.cellid, faces.name )

    #update the halo values
    if size > 1 :
        w_halosend = meshpart.define_halosend(w, w_halosend, faces.cellid, indSend, mystruct)
        w_halorecv = meshpart.all_to_all(w_halosend, taille, mystruct, variables,
                                                   scount, sdepl, rcount, rdepl)
        w_halo = meshpart.halo_value(w_halo, faces.cellid, w_halorecv, indRecv, mystruct)

    #update the rezidus using explicit scheme
    rezidus = meshpart.ExplicitScheme(w, w_ghost, w_halo, faces.cellid, faces.normal,
                                      faces.name, mystruct, variables)


    #update the new solution
    wn = meshpart.update(w, wn, dt, rezidus, cells.volume)

    #save vtk files for the solution
    if(n%T == 0):
        meshpart.save_paraview_results(w, n, m, t,dt, rank , size, cells.nodeid, nodes.vertex)
        m += 1


    w = wn
    n += 1
#
stop = timeit.default_timer()

if rank == 0 : print(stop - start)
