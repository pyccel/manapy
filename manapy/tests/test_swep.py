#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:51:37 2020

@author: kissami
"""
import timeit
import os
import numpy as np
from mpi4py import MPI
from manapy import ddm


# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']

except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR, '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def test_swep():

    if RANK == 0:
        #reading gmsh file and partitioning into size subdomains
        filename = os.path.join(MESH_DIR, "meshpaper2007.msh")
        ddm.meshpart(SIZE, filename)
        #removing existing vtk files
        mypath = "results"
        if not os.path.exists(mypath):
            os.mkdir(mypath)
        for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
    COMM.Barrier()

    start = timeit.default_timer()

    #generating local grid for each subdomain
    grid = {}
    grid = ddm.generate_mesh()

    faces = grid["faces"]
    cells = grid["cells"]
    halos = grid["halos"]
    nodes = grid["nodes"]


    nbelements = len(cells.center)
    nbfaces = len(faces.name)
    #nbnodes = len(nodes.vertex)

    variables = tuple(['h', 'hu', 'hv', 'hc', 'Z'])
    mystruct = np.dtype([('h', np.float64),
                         ('hu', np.float64),
                         ('hv', np.float64),
                         ('hc', np.float64),
                         ('Z', np.float64),])

    w_c = np.recarray(nbelements, dtype=mystruct)
    w_x = np.recarray(nbelements, dtype=mystruct)
    w_y = np.recarray(nbelements, dtype=mystruct)
    source = np.recarray(nbelements, dtype=mystruct)
    w_ghost = np.recarray(nbfaces, dtype=mystruct)


    #compute the arrays needed for the mpi communication
    scount, sdepl, rcount, rdepl, taille, indsend = ddm.prepare_comm(cells, halos)

    w_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    wx_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    wy_halosend = np.zeros(len(halos.halosint), dtype=mystruct)

    w_n = w_c
    ###Initialisation
    w_c = ddm.initialisation(w_c, cells.center)


    if RANK == 0: print("Start Computation")
    cfl = 0.5
    ####calculation of the time step
    d_t = ddm.time_step(w_c, cfl, faces.normal, cells.volume, cells.faceid)
    dt_i = np.zeros(1)
    COMM.Allreduce(d_t, dt_i, MPI.MIN)
    d_t = np.float64(dt_i)
    
    time = 0
    tfinal = 5000
    #order = 3 #(1 : first order, 2: van albada, 3: barth jeperson)

    #saving 25 vtk file
    tot = int(tfinal/d_t/25)
    miter = 0
    niter = 0
    order = 1
    #loop over time
    while time < tfinal:

        time = time + d_t

        #update the ghost values for the boundary conditions
        w_ghost = ddm.ghost_value(w_c, w_ghost, faces.cellid, faces.name, faces.normal)

        #update the halo values
        #if SIZE > 1:
        w_halosend = ddm.define_halosend(w_c, w_halosend, indsend)
        w_halo = ddm.all_to_all(w_halosend, taille, mystruct, variables,
                                scount, sdepl, rcount, rdepl)

        #compute derivative
        w_x, w_y = ddm.derivxy(w_c, w_ghost, w_halo, cells.center, halos.centvol,
                               nodes.cellid, nodes.halonid, cells.nodeid, w_x, w_y)

        #update the halo  derivatives values
        wx_halosend = ddm.define_halosend(w_x, wx_halosend, indsend)
        wy_halosend = ddm.define_halosend(w_y, wy_halosend, indsend)

        wx_halo = ddm.all_to_all(wx_halosend, taille, mystruct, variables,
                                 scount, sdepl, rcount, rdepl)
        wy_halo = ddm.all_to_all(wy_halosend, taille, mystruct, variables,
                                 scount, sdepl, rcount, rdepl)

        #update the rezidus using explicit scheme
        rezidus = ddm.explicitscheme(w_c, w_x, w_y, w_ghost, w_halo, wx_halo, wy_halo,
                                     faces.cellid, cells.faceid, cells.center, halos.centvol,
                                     faces.center, faces.normal, faces.halofid,
                                     faces.name, mystruct, order)
        #source = ddm.term_source(w_c, w_ghost, cells.nodeid, cells.faceid, cells.center, faces.cellid,
        #                         faces.nodeid, faces.normal, faces.center, nodes.vertex, mystruct)


        #update the new solution
        w_n = ddm.update(w_c, w_n, d_t, rezidus, cells.volume)

        #save vtk files for the solution
        if niter%tot == 0:
            ddm.save_paraview_results(w_c, niter, miter, time, d_t, RANK, SIZE,
                                      cells.nodeid, nodes.vertex)
            miter += 1

        w_c = w_n
        niter += 1

        ####calculation of the time step
        d_t = ddm.time_step(w_c, cfl, faces.normal, cells.volume, cells.faceid)
        dt_i = np.zeros(1)
        COMM.Allreduce(d_t, dt_i, MPI.MIN)
        d_t = np.float64(dt_i)
        #saving 25 vtk file
        tot = int(tfinal/d_t/25)

    stop = timeit.default_timer()

    if RANK == 0: print(stop - start)

test_swep()
