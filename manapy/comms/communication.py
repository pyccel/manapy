#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:41:06 2021

@author: kissami
"""
import numpy as np
from mpi4py import MPI
from numba import njit

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

@njit
def define_halosend(w_c:'float[:]', w_halosend:'float[:]', indsend:'int[:]'):
    
    w_halosend[:] = w_c[indsend[:]]

def create_mpi_graph(neighbors):
    topo = COMM.Create_dist_graph_adjacent(neighbors, neighbors,
                                           sourceweights=None, destweights=None)
    return topo


def all_to_all(w_halosend, taille, scount, rcount, w_halorecv, comm_ptr):

    s_msg = r_msg = 0
    s_msg = [w_halosend, (scount), MPI.DOUBLE_PRECISION]
    r_msg = [w_halorecv, (rcount), MPI.DOUBLE_PRECISION]

    comm_ptr.Neighbor_alltoallv(s_msg, r_msg)

    w_halorecv = r_msg[0]


def prepare_comm(cells, halos):

    scount = np.zeros(SIZE, dtype=np.intc)
    rcount = np.zeros(SIZE, dtype=np.intc)
    taille = 0
    
    if SIZE > 1:
        for i in range(len(halos._neigh[0])):
            scount[halos._neigh[0][i]] = halos._neigh[1][i]

        rcount = COMM.alltoall(scount)

        for i in range(SIZE):
            taille += rcount[i]

        taille = int(taille)

        indsend = np.zeros(0, dtype=np.int64)
        for i in range(len(halos._halosint)):
            indsend = np.append(indsend, cells._globtoloc[halos._halosint[i]])
            
    else:
        indsend = np.zeros(1, dtype=np.int64)

    rcount = np.asarray(rcount)
    
    if SIZE == 1:
        comm_ptr = create_mpi_graph([0])
    else:
        comm_ptr = create_mpi_graph(halos._neigh[0])
        
        scount = scount[scount>0]
        rcount = rcount[rcount>0]
    
    return scount, rcount, indsend, taille, comm_ptr


def update_haloghost_info_2d(nodes, cells, halos, nbnodes, halonodes):
    
    ghostcenter = {}
    ghostfaceinfo = {}
    
    scount_node = np.zeros(SIZE, dtype=np.int64)
    sdepl_node = np.zeros(SIZE, dtype=np.int64)
    
    rcount_node = np.zeros(SIZE, dtype=np.int64)
    rdepl_node = np.zeros(SIZE, dtype=np.int64)
    taille = 0
    
    taille_node_ghost = np.zeros(SIZE, dtype=np.int64)
    count1 = 6
    count2 = 4
    
    import collections
    if SIZE > 1:
        for i in halonodes:
            for j in range(len(nodes._ghostcenter[i])):
                for k in nodes._nparts[nodes._loctoglob[i]]:
                    if k != RANK:
                        ghostcenter.setdefault(k, []).append([
                                nodes._loctoglob[i], 
                                nodes._ghostcenter[i][j][0],
                                nodes._ghostcenter[i][j][1],
                                cells._loctoglob[nodes._ghostcenter[i][j][2]],
                                nodes._ghostcenter[i][j][3], 
                                nodes._ghostcenter[i][j][4]])
                        
                        ghostfaceinfo.setdefault(k, []).append([
                                        nodes._ghostfaceinfo[i][j][0],
                                        nodes._ghostfaceinfo[i][j][1],
                                        nodes._ghostfaceinfo[i][j][2], 
                                        nodes._ghostfaceinfo[i][j][3]])
    
                        taille_node_ghost[k] += 1

        ghostcenter = collections.OrderedDict(sorted(ghostcenter.items()))
        ghostfaceinfo = collections.OrderedDict(sorted(ghostfaceinfo.items()))

        
        for i in range(len(halos._neigh[0])):
            scount_node[halos._neigh[0][i]] = taille_node_ghost[halos._neigh[0][i]]
        
        for i in range(1, SIZE):
            sdepl_node[i] = sdepl_node[i-1] + scount_node[i-1]
        
        rcount_node = COMM.alltoall(scount_node)
        
        for i in range(1, SIZE):
            rdepl_node[i] = rdepl_node[i-1] + rcount_node[i-1]
        
        for i in range(SIZE):
            taille += rcount_node[i]
    
    #######################Ghost center info##################################
        sendbuf1 = []
        
        for i, j in ghostcenter.items():
            sendbuf1.extend(j)
        
        sendbuf1 = np.asarray(sendbuf1)
        
        ghostcenter_halo = np.ones((taille, count1))
        
        type_ligne = MPI.DOUBLE.Create_contiguous(count1)
        type_ligne.Commit()
        
        s_msg = r_msg = 0
        s_msg = [sendbuf1, (scount_node, sdepl_node), type_ligne]
        r_msg = [ghostcenter_halo, (rcount_node, rdepl_node), type_ligne]
        
        COMM.Alltoallv(s_msg, r_msg)
        type_ligne.Free()
        
        recvbuf1 = {}
        for i in range(len(ghostcenter_halo)):
            recvbuf1.setdefault(ghostcenter_halo[i][0], []).append([ghostcenter_halo[i][1], ghostcenter_halo[i][2], 
                                                                    ghostcenter_halo[i][3], ghostcenter_halo[i][4],
                                                                    ghostcenter_halo[i][5]])
        for i in halonodes:
            if recvbuf1.get(nodes._loctoglob[i]):
                nodes._haloghostcenter[i].extend(recvbuf1[nodes._loctoglob[i]])
                
                
    #########################Ghost face info####################################
        sendbuf2 = []
            
        for i, j in ghostfaceinfo.items():
            sendbuf2.extend(j)
        
        sendbuf2 = np.asarray(sendbuf2)
        
        ghostfaceinfo_halo = np.ones((taille, count2))
        
        type_ligne = MPI.DOUBLE.Create_contiguous(count2)
        type_ligne.Commit()
        
        s_msg = r_msg = 0
        s_msg = [sendbuf2, (scount_node, sdepl_node), type_ligne]
        r_msg = [ghostfaceinfo_halo, (rcount_node, rdepl_node), type_ligne]
        
        COMM.Alltoallv(s_msg, r_msg)
        type_ligne.Free()
        
        recvbuf2 = {}
        for i in range(len(ghostfaceinfo_halo)):
            recvbuf2.setdefault(ghostcenter_halo[i][0], []).append([ghostfaceinfo_halo[i][0], ghostfaceinfo_halo[i][1], 
                                                                    ghostfaceinfo_halo[i][2], ghostfaceinfo_halo[i][3]])
        for i in halonodes:
            if recvbuf2.get(nodes._loctoglob[i]):
                nodes._haloghostfaceinfo[i].extend(recvbuf2[nodes._loctoglob[i]])
    
    ###########After communication#############################################
    maxGhostCell = 0
    for i in range(nbnodes):
        maxGhostCell = max(maxGhostCell, len(nodes._ghostcenter[i]))
    
    for i in range(nbnodes):
        iterator = maxGhostCell - len(nodes._ghostcenter[i])
        for k in range(iterator):
            nodes._ghostcenter[i].append([-1., -1., -1. , -1., -1])
            nodes._ghostfaceinfo[i].append([-1., -1., -1. , -1.])
    
        if len(nodes._ghostcenter[i]) == 0 :
                nodes._ghostcenter[i].append([-1, -1., -1., -1., -1])
                nodes._ghostfaceinfo[i].append([-1., -1., -1. , -1.])
    
    maxhaloGhostCell = 0
    for i in range(nbnodes):
        maxhaloGhostCell = max(maxhaloGhostCell, len(nodes._haloghostcenter[i]))
        
    for i in range(nbnodes):
        iterator = maxhaloGhostCell - len(nodes._haloghostcenter[i])
        for k in range(iterator):
            nodes._haloghostcenter[i].append([-1., -1., -1., -1, -1])
            nodes._haloghostfaceinfo[i].append([-1., -1., -1. , -1.])
        
        if len(nodes._haloghostcenter[i]) == 0 :
            nodes._haloghostcenter[i].append([-1, -1., -1., -1, -1])   
            nodes._haloghostfaceinfo[i].append([-1., -1., -1. , -1.])
        
            
    #local halo index of haloext
    haloexttoind = {}
    for i in halonodes:
        for j in range(nodes._halonid[i][-1]):
            haloexttoind[halos._halosext[nodes._halonid[i][j]][0]] = nodes._halonid[i][j]
    # maxsize = 0
    cmpt = 0
    for i in halonodes:
        # if nodes._name[i] == 10:
        for j in range(len(nodes._haloghostcenter[i])):
            if nodes._haloghostcenter[i][j][-1] != -1:
                nodes._haloghostcenter[i][j][-3] = haloexttoind[int(nodes._haloghostcenter[i][j][-3])]
                nodes._haloghostcenter[i][j][-1] = cmpt
                cmpt = cmpt + 1
            
    maxsize = cmpt
    
    nodes._ghostcenter       = np.asarray(nodes._ghostcenter)
    nodes._haloghostcenter   = np.asarray(nodes._haloghostcenter)
    nodes._ghostfaceinfo     = np.asarray(nodes._ghostfaceinfo)
    nodes._haloghostfaceinfo = np.asarray(nodes._haloghostfaceinfo)
    
    return maxsize

def update_haloghost_info_3d(nodes, cells, halos, nbnodes, halonodes):
    
    ghostcenter = {}
    ghostfaceinfo = {}
    
    scount_node = np.zeros(SIZE, dtype=np.int64)
    sdepl_node = np.zeros(SIZE, dtype=np.int64)
    
    rcount_node = np.zeros(SIZE, dtype=np.int64)
    rdepl_node = np.zeros(SIZE, dtype=np.int64)
    taille = 0
    
    taille_node_ghost = np.zeros(SIZE, dtype=np.int64)
    count1 = 7
    count2 = 6
    
    import collections
    maxsize = 0
    if SIZE > 1:
        for i in halonodes:
            for j in range(len(nodes._ghostcenter[i])):
                for k in nodes._nparts[nodes._loctoglob[i]]:
                    if k != RANK:
                        ghostcenter.setdefault(k, []).append([
                            nodes._loctoglob[i], 
                            nodes._ghostcenter[i][j][0],
                            nodes._ghostcenter[i][j][1], 
                            nodes._ghostcenter[i][j][2], 
                            cells._loctoglob[nodes._ghostcenter[i][j][3]],
                            nodes._ghostcenter[i][j][4], 
                            nodes._ghostcenter[i][j][5]])
                        
                        
                        ghostfaceinfo.setdefault(k, []).append([
                                            nodes._ghostfaceinfo[i][j][0],
                                            nodes._ghostfaceinfo[i][j][1],
                                            nodes._ghostfaceinfo[i][j][2], 
                                            nodes._ghostfaceinfo[i][j][3],
                                            nodes._ghostfaceinfo[i][j][4],
                                            nodes._ghostfaceinfo[i][j][5]])
                        taille_node_ghost[k] += 1
        
        ghostcenter = collections.OrderedDict(sorted(ghostcenter.items()))
        ghostfaceinfo = collections.OrderedDict(sorted(ghostfaceinfo.items()))

        
        for i in range(len(halos._neigh[0])):
            scount_node[halos._neigh[0][i]] = taille_node_ghost[halos._neigh[0][i]]
        
        for i in range(1, SIZE):
            sdepl_node[i] = sdepl_node[i-1] + scount_node[i-1]
        
        rcount_node = COMM.alltoall(scount_node)
        
        for i in range(1, SIZE):
            rdepl_node[i] = rdepl_node[i-1] + rcount_node[i-1]
        
        for i in range(SIZE):
            taille += rcount_node[i]
    
        sendbuf1 = []
        
        for i, j in ghostcenter.items():
            sendbuf1.extend(j)
            
        sendbuf1 = np.asarray(sendbuf1)
        
        ghostcenter_halo = np.ones((taille, count1))
        
        
        type_ligne = MPI.DOUBLE.Create_contiguous(count1)
        type_ligne.Commit()
        
        s_msg = r_msg = 0
        s_msg = [sendbuf1, (scount_node, sdepl_node), type_ligne]
        r_msg = [ghostcenter_halo, (rcount_node, rdepl_node), type_ligne]
        
        COMM.Alltoallv(s_msg, r_msg)
        type_ligne.Free()
        
        recvbuf1 = {}
        for i in range(len(ghostcenter_halo)):
            recvbuf1.setdefault(ghostcenter_halo[i][0], []).append([ghostcenter_halo[i][1], ghostcenter_halo[i][2], 
                                                                    ghostcenter_halo[i][3], ghostcenter_halo[i][4],
                                                                    ghostcenter_halo[i][5], ghostcenter_halo[i][6]])
        for i in halonodes:
            if recvbuf1.get(nodes._loctoglob[i]):
                nodes._haloghostcenter[i].extend(recvbuf1[nodes._loctoglob[i]])
    
    
    
    ############################Face info ###################################################
    
        sendbuf2 = []
        
        for i, j in ghostfaceinfo.items():
            sendbuf2.extend(j)
            
        sendbuf2 = np.asarray(sendbuf2)
        
        ghostfaceinfo_halo = np.ones((taille, count2))
        
        
        type_ligne = MPI.DOUBLE.Create_contiguous(count2)
        type_ligne.Commit()
        
        s_msg = r_msg = 0
        s_msg = [sendbuf2, (scount_node, sdepl_node), type_ligne]
        r_msg = [ghostfaceinfo_halo, (rcount_node, rdepl_node), type_ligne]
        
        COMM.Alltoallv(s_msg, r_msg)
        type_ligne.Free()
        
        recvbuf2 = {}
        for i in range(len(ghostfaceinfo_halo)):
            recvbuf2.setdefault(ghostcenter_halo[i][0], []).append([ghostfaceinfo_halo[i][0], ghostfaceinfo_halo[i][1], 
                                                                    ghostfaceinfo_halo[i][2], ghostfaceinfo_halo[i][3],
                                                                    ghostfaceinfo_halo[i][4], ghostfaceinfo_halo[i][5]])
        for i in halonodes:
            if recvbuf2.get(nodes._loctoglob[i]):
                nodes._haloghostfaceinfo[i].extend(recvbuf2[nodes._loctoglob[i]])
                
                
    ###############End communications########################################################
    
    
    maxGhostCell = 0
    for i in range(nbnodes):
        maxGhostCell = max(maxGhostCell, len(nodes._ghostcenter[i]))
    
    for i in range(nbnodes):
        iterator = maxGhostCell - len(nodes._ghostcenter[i])
        for k in range(iterator):
            nodes._ghostcenter[i].append([-1., -1., -1., -1., -1., -1.])
            nodes._ghostfaceinfo[i].append([-1., -1., -1. , -1., -1, -1])
    
        if len(nodes._ghostcenter[i]) == 0 :
                nodes._ghostcenter[i].append([-1, -1., -1.,-1., -1., -1.])
                nodes._ghostfaceinfo[i].append([-1., -1., -1. , -1., -1, -1])
                
    maxGhostCell = 0
    for i in range(nbnodes):
        maxGhostCell = max(maxGhostCell, len(nodes._haloghostcenter[i]))
        
    for i in range(nbnodes):
        iterator = maxGhostCell - len(nodes._haloghostcenter[i])
        for k in range(iterator):
            nodes._haloghostcenter[i].append([-1., -1., -1., -1., -1., -1.])
            nodes._haloghostfaceinfo[i].append([-1., -1., -1. , -1., -1, -1])
    
        if len(nodes._haloghostcenter[i]) == 0 :
                nodes._haloghostcenter[i].append([-1, -1., -1.,-1., -1., -1.])
                nodes._haloghostfaceinfo[i].append([-1., -1., -1. , -1., -1, -1])
                
    #local halo index of haloext
    haloexttoind = {}
    for i in halonodes:
        for j in range(nodes._halonid[i][-1]):
            haloexttoind[halos._halosext[nodes._halonid[i][j]][0]] = nodes._halonid[i][j]
    cmpt = 0
    for i in halonodes:
        for j in range(len(nodes._haloghostcenter[i])):
            if nodes._haloghostcenter[i][j][-1] != -1:
                nodes._haloghostcenter[i][j][-3] = haloexttoind[int(nodes._haloghostcenter[i][j][-3])]
                nodes._haloghostcenter[i][j][-1] = cmpt
                cmpt = cmpt + 1
    maxsize = cmpt#int(max(maxsize, nodes._haloghostcenter[i][j][-1]+1))
    
  
        
    
    nodes._ghostcenter     = np.asarray(nodes._ghostcenter)
    nodes._haloghostcenter = np.asarray(nodes._haloghostcenter)
    nodes._ghostfaceinfo     = np.asarray(nodes._ghostfaceinfo)
    nodes._haloghostfaceinfo = np.asarray(nodes._haloghostfaceinfo)
    
    return maxsize
