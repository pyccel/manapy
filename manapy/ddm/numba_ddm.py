#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:15:29 2021

@author: kissami
"""
from numpy import zeros, asarray, double, int64, dot, cross, append, where, array, unique, float32
from collections import OrderedDict
from numba import njit
import os

@njit
def create_npart_cpart(cell_nodeid:'int[:,:]', npart:'int[:]', epart:'int[:]', nbnodes:'int', nbelements:'int',
                       SIZE:'int', dim:'int'):
    npart       = [ [i ] for i in npart ]
    cpart       = [ [i ] for i in epart ]
    neighsub    = [[i for i in range(0)]  for i in range(SIZE)]
    cell_part   = [[i for i in range(0)]  for i in range(SIZE)]
    node_part   = [[i for i in range(0)]  for i in range(SIZE)]
    halo_cellid = [[i for i in range(0)]  for i in range(SIZE)]

    for i in range(nbelements):
        for j in range(dim+1):
            k = cell_nodeid[i][j]
            if epart[i] not in npart[k]:
                npart[k].append(epart[i])
            node_part[epart[i]].append(k)
        cell_part[epart[i]].append(i)
    
    
    for i in range(nbelements):
        for j in range(dim+1):
            for k in range(len(npart[cell_nodeid[i][j]])):
                if npart[cell_nodeid[i][j]][k] not in cpart[i]:
                    cpart[i].append(npart[cell_nodeid[i][j]][k])
                    
    for i in range(nbnodes):
        if len(npart[i]) > 1:
            for j in range(len(npart[i])):
                neighsub[npart[i][j]].extend(npart[i])
    
    for i in range(SIZE):
        for j in cell_part[i]:
            a = 0.
            for k in range(dim+1):
                a += len(npart[cell_nodeid[j][k]])
                if a > dim+1:
                    halo_cellid[i].append(j)
                
    return npart, cpart, neighsub, node_part, cell_part, halo_cellid

def unique_func(array):
    uniq, index = unique(array, return_index=True)
    return uniq[index.argsort()]

@njit(fastmath=True)
def wedge_3d(u, v):
    wedge = zeros(3)
    
    wedge[0] = u[1]*v[2] - u[2]*v[1]
    wedge[1] = u[2]*v[0] - u[0]*v[2]
    wedge[2] = u[0]*v[1] - u[1]*v[0]
    
    return wedge

@njit(fastmath=True)
def dot_vec3(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

@njit(fastmath=True)
def det_vec_3d(u, v, w):
    return dot_vec3(u, wedge_3d(v, w))

@njit(fastmath=True)
def oriente_3dfacenodeid(nodeid:'int[:,:]', normal:'double[:,:]', vertex:'double[:,:]'):
    
    nbfaces = len(nodeid)
    v1 = zeros(3)
    v2 = zeros(3)
    
    for i in range(nbfaces):
        n1 = nodeid[i][0]; n2 = nodeid[i][1]; n3 = nodeid[i][2]
        s1 = vertex[n1][0:3]; s2 = vertex[n2][0:3]; s3 = vertex[n3][0:3];
        
        v1[:] = s2[:] - s1[:] 
        v2[:] = s3[:] - s1[:]
        
        if dot(cross(v1, v2), normal[i]) < 0:
            nodeid[i][1] = n3; nodeid[i][2] = n2
            
    return nodeid

@njit 
def create_NeighborCellByFace(faceid:'int[:,:]', cellid:'int[:,:]', nbelements:'int', dim:'int'):
    
    cellfid = [[i for i in range(0)] for i in range(nbelements)]
    #Création des 3/4 triangles voisins par face
    for i in range(nbelements):
        for j in range(dim+1):
            f = faceid[i][j]
            if cellid[f][1] != -1:
                if i == cellid[f][0]:
                    cellfid[i].append(cellid[f][1])
                else:
                    cellfid[i].append(cellid[f][0])
            else:
                cellfid[i].append(-1)
    cellfid = asarray(cellfid, dtype=int64)
    
    return cellfid

@njit 
def create_2doppNodeOfFaces(nodeidc:'int[:,:]', faceidc:'int[:,:]', nodeidf:'int[:,:]', nbelements:'int', nbfaces:'int'):
    #    #TODO improve the opp node creation
    oppnodeid = [[i for i in range(0)] for i in range(nbfaces)]
   
    for i in range(nbelements):
        f1 = faceidc[i][0]; f2 = faceidc[i][1]; f3 = faceidc[i][2]
        n1 = nodeidc[i][0]; n2 = nodeidc[i][1]; n3 = nodeidc[i][2] 
       
        if n1 not in nodeidf[f1] :
            oppnodeid[f1].append(n1)
        if n1 not in nodeidf[f2] :
            oppnodeid[f2].append(n1)
        if n1 not in nodeidf[f3] :
            oppnodeid[f3].append(n1)
        
        if n2 not in nodeidf[f1] :
            oppnodeid[f1].append(n2)
        if n2 not in nodeidf[f2] :
            oppnodeid[f2].append(n2)
        if n2 not in nodeidf[f3] :
            oppnodeid[f3].append(n2)
        
        if n3 not in nodeidf[f1] :
            oppnodeid[f1].append(n3)
        if n3 not in nodeidf[f2] :
            oppnodeid[f2].append(n3)
        if n3 not in nodeidf[f3] :
            oppnodeid[f3].append(n3)
        
    for i in range(nbfaces):
        if len(oppnodeid[i]) < 2:
            oppnodeid[i].append(-1)
    
    oppnodeid = asarray(oppnodeid, dtype=int64)      
            
    return oppnodeid

@njit 
def create_3doppNodeOfFaces(nodeidc:'int[:,:]', faceidc:'int[:,:]', nodeidf:'int[:,:]', nbelements:'int', nbfaces:'int'):
    
    #TODO improve the opp node creation
    oppnodeid = [[i for i in range(0)] for i in range(nbfaces)]
    for i in range(nbelements):
        f1 = faceidc[i][0]; f2 = faceidc[i][1]; f3 = faceidc[i][2]; f4 = faceidc[i][3]; 
        n1 = nodeidc[i][0]; n2 = nodeidc[i][1]; n3 = nodeidc[i][2]; n4 = nodeidc[i][3]; 
        
        if n1 not in nodeidf[f1] :
            oppnodeid[f1].append(n1)
        if n1 not in nodeidf[f2] :
            oppnodeid[f2].append(n1)
        if n1 not in nodeidf[f3] :
            oppnodeid[f3].append(n1)
        if n1 not in nodeidf[f4] :
            oppnodeid[f4].append(n1)
        
        if n2 not in nodeidf[f1] :
            oppnodeid[f1].append(n2)
        if n2 not in nodeidf[f2] :
            oppnodeid[f2].append(n2)
        if n2 not in nodeidf[f3] :
            oppnodeid[f3].append(n2)
        if n2 not in nodeidf[f4] :
            oppnodeid[f4].append(n2)
        
        if n3 not in nodeidf[f1] :
            oppnodeid[f1].append(n3)
        if n3 not in nodeidf[f2] :
            oppnodeid[f2].append(n3)
        if n3 not in nodeidf[f3] :
            oppnodeid[f3].append(n3)
        if n3 not in nodeidf[f4] :
            oppnodeid[f4].append(n3)
        
        if n4 not in nodeidf[f1] :
            oppnodeid[f1].append(n4)
        if n4 not in nodeidf[f2] :
            oppnodeid[f2].append(n4)
        if n4 not in nodeidf[f3] :
            oppnodeid[f3].append(n4)
        if n4 not in nodeidf[f4] :
            oppnodeid[f4].append(n4)
        
            
    for i in range(nbfaces):
        if len(oppnodeid[i]) < 2:
            oppnodeid[i].append(-1)
            
    oppnodeid = asarray(oppnodeid, dtype=int64)   
    
    return oppnodeid


@njit 
def create_node_cellid(nodeid:'int[:,:]', vertex:'double[:,:]', nbelements:'int', nbnodes:'int', dim:'int'):
    
    tmp = [[i for i in range(0)] for i in range(nbnodes)]
    longn = zeros(nbnodes, dtype=int64)
    
    for i in range(nbelements):
        for j in range(dim+1):
            tmp[nodeid[i][j]].append(i)
            longn[nodeid[i][j]] = longn[nodeid[i][j]] + 1
    
    longc = zeros(nbelements, dtype=int64)
    tmp2 = [[i for i in range(0)] for i in range(nbelements)]
    
    for i in range(nbelements):
        for j in range(dim+1):
            for k in range(len(tmp[nodeid[i][j]])):
                if (tmp[nodeid[i][j]][k] not in tmp2[i] and  tmp[nodeid[i][j]][k] != i):
                    tmp2[i].append(tmp[nodeid[i][j]][k])
                    longc[i] = longc[i] + 1
        tmp2[i].sort()
   
    maxlongn = int(max(longn))
    cellid = [[-1 for i in range(maxlongn)] for i in range(nbnodes)]
    
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            cellid[i][j] = tmp[i][j]
            
    for i in range(nbnodes):
        cellid[i].append(longn[i])
    cellid = asarray(cellid, dtype=int64)
    
    maxlongc = int(max(longc))
    cellnid = [[-1 for i in range(maxlongc)] for i in range(nbelements)]
    
    for i in range(len(tmp2)):
        for j in range(len(tmp2[i])):
            cellnid[i][j] = tmp2[i][j]
    
    for i in range(nbelements):
        cellnid[i].append(longc[i])
   
    cellnid = asarray(cellnid, dtype=int64)   
    
    
    return cellid, cellnid


def read_mesh_file(size, rank):
    
    MESH_DIR = "meshes"+str(size)+"PROC"
    filename = os.path.join(MESH_DIR, 'mesh'+str(rank)+'.txt')
    file = open(filename)
    
    tc = []; nodeid = []; vertex = []
    halosint = []; halosext = []; centvol = []; 
    Cellloctoglob = []; Cellglobtoloc = []; neigh = []
    Nodeloctoglob = []; nparts = []; neigh = []
    Cellglobtoloc = OrderedDict()
    #Lecture des cellules à partir du fichier mesh..txt
    if size > 1 and rank == 0:
        for line in file:
            if line == "GtoL\n":
                continue
            if line == "endGtoL\n":
                continue
            if line == "elements\n":
                break
            tc.append([int64(x) for x in line.split()][0])
            
    #Lecture des coordonnées des noeuds à partir du fichier mesh..txt
    for line in file:
        #read elements
        if line == "elements\n":
            continue
        if line == "endelements\n":
            continue
        if line == "nodes\n":
            break
        nodeid.append([int64(x) for x in line.split()])
    #Lecture des coordonnées des noeuds à partir du fichier mesh..txt
    for line in file:
        #read nodes
        if line == "nodes\n":
            continue
        if line == "endnodes\n":
            continue
        if line == "halosint\n":
            break
        vertex.append([double(x) for x in line.split()])
        
    nbcells = len(nodeid)
    if size == 1:
         tc = zeros(nbcells, dtype=int64)
         #tc to gather solution of linear system
         for i in range(nbcells):
             tc[i] = i
             
    if size > 1:
        
        filename_master = os.path.join(MESH_DIR, 'mesh_master.txt')
        file_master = open(filename_master)
        
        for line in file:
            if "endhalosint" in line:
                break
            halosint.append(int(line))
    
        for line in file:
            # process the line
            if "halosext" in line:
                continue
            if "centvol" in line:
                break
            halosext.append([int(x) for x in line.split()])
    
        for line in file:
            if "centvol" in line:
                continue
            if "globalcelltolocal" in line:
                break
            centvol.append([float(x) for x in line.split()])
    
        cmpt = 0
        for line in file:
        #read Global cell to local
            if line == "globalcelltomocal\n":
                continue
            if line == "endglobalcelltolocal\n":
                break
            Cellglobtoloc[int(line)] = cmpt
            Cellloctoglob.append(int(line))
            cmpt += 1
    
        cmpt = 0
        for line in file:
            #read Local Node To Global
            if line == "localnodetoglobal\n":
                continue
            if line == "endlocalnodetoglobal\n":
                break
            Nodeloctoglob.append(int(line))
            cmpt += 1
        for line in file:
            #read LocalToGlobal
            if line == "neigh\n":
                continue
            if line == "endneigh\n":
                break
            neigh.append([int64(x) for x in line.split()])
    
        for line in file_master:
            #read LocalToGlobal
            if line == "nodeparts\n":
                continue
            if line == "endnodeparts\n":
                break
            nparts.append([int64(x) for x in line.split()])
        
    return tc, nodeid, vertex, halosint, halosext, centvol, Cellglobtoloc, Cellloctoglob, Nodeloctoglob, neigh, nparts


def create_2d_halo_structure(cells, faces, nodes, halos, size, nbcells, nbfaces, nbnodes):

    facenameoldf = faces._name
    
    #TODO change adding cell index
    faces._halofid = zeros(nbfaces, dtype=int64)
    halos._nodes = zeros(nbnodes, dtype=int64)
    halos._faces = OrderedDict()
    
    if size > 1:
        k = 1
        for i in range(len(halos._halosext)):
            halos._faces[tuple([halos._halosext[i][1], halos._halosext[i][2]])] = k
            halos._faces[tuple([halos._halosext[i][2], halos._halosext[i][3]])] = k + 1
            halos._faces[tuple([halos._halosext[i][1], halos._halosext[i][3]])] = k + 2
            k = k+3

        for i in range(nbfaces):
            n1 = nodes._loctoglob[faces._nodeid[i][0]]
            n2 = nodes._loctoglob[faces._nodeid[i][1]]
            if  halos._faces.get(tuple([n1, n2])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n1, n2])))/3)

            if halos._faces.get(tuple([n2, n1])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10

                faces._halofid[i] = int((-1+halos._faces.get(tuple([n2, n1])))/3)
                
        longueur = 0
        longh = []
        tmp = [[] for i in range(nbnodes)]
        for i in range(nbnodes):
            if nodes._name[i] == 10:
                halos._nodes[i] = nodes._loctoglob[i]
                arg = where(halos._halosext[:,1:4] == nodes._loctoglob[i])
                tmp[i].append(arg[0])
                longueur = max(longueur, len(arg[0]))
                longh.append(len(arg[0]))
            else:
                tmp[i].append(array([-1]))
                longh.append(0)

        nodes._halonid = [[-1]*longueur for i in range(nbnodes)]
        
        for i in range(len(tmp)):
            for j in range(len(tmp[i][0])):
                nodes._halonid[i][j] = tmp[i][0][j]
                
        for i in range(len(nodes._halonid)):
            nodes._halonid[i].append(longh[i])
                
    if size == 1 : 
        nodes._halonid = zeros((nbnodes,2), dtype=int64)
        halos._centvol  = zeros((2,2))
        halos._halosint = zeros((2,2))
        
      # A vérifier !!!!!!
    faces._ghostcenter = zeros((nbfaces, 3))
    nodes._ghostcenter = [[] for i in range(nbnodes)]
    nodes._ghostfaceinfo   = [[] for i in range(nbnodes)]
    
    nodes._haloghostcenter = [[] for i in range(nbnodes)]
    nodes._haloghostfaceinfo = [[] for i in range(nbnodes)]
    
    #compute the ghostcenter for each face and each node
    for i in range(nbfaces):
        nod1 = faces._nodeid[i][1]
        nod2 = faces._nodeid[i][0]
        if faces._name[i] == 1 or  faces._name[i] == 2 or faces._name[i] ==  3 or faces._name[i] == 4:
            
            x_1 = nodes._vertex[nod1]
            x_2 = nodes._vertex[nod2]
            
            c_left = faces._cellid[i][0]
            v_1 = cells._center[c_left]
            gamma = ((v_1[0] - x_2[0])*(x_1[0]-x_2[0]) + (v_1[1]-x_2[1])*(x_1[1]-x_2[1]))/((x_1[0]-x_2[0])**2 + (x_1[1]-x_2[1])**2)
           
            kk = array([gamma * x_1[0] + (1 - gamma) * x_2[0], gamma * x_1[1] + (1 - gamma) * x_2[1]])
            
            v_2 = array([2 * kk[0] + ( -1 * v_1[0]), 2 * kk[1] + ( -1 * v_1[1])])

            faces._ghostcenter[i] = [v_2[0], v_2[1], gamma]
            nodes._ghostcenter[nod1].append([v_2[0], v_2[1], faces._cellid[i][0], facenameoldf[i], i])
            nodes._ghostcenter[nod2].append([v_2[0], v_2[1], faces._cellid[i][0], facenameoldf[i], i])
            
            ll = [faces._center[i][0], faces._center[i][1], faces._normal[i][0], faces._normal[i][1]]
            nodes._ghostfaceinfo[nod1].append(ll)
            nodes._ghostfaceinfo[nod2].append(ll)
            
        else:
            faces._ghostcenter[i] = [-1., -1., -1.]
            
    #define halo cells neighbor by nodes
    maxhalonid = 0
    cells._halonid = [[] for i in range(nbcells)]
    for i in range(nbcells):
        for j in range(3):
            nod = cells._nodeid[i][j]
            k = nodes._halonid[nod][-1]
            cells._halonid[i].extend(nodes._halonid[nod][:k])
        cells._halonid[i] = list(set(cells._halonid[i]))
        maxhalonid = max(maxhalonid, len(cells._halonid[i]))
    
    for i in range(nbcells):
        numb = len(cells._halonid[i])
        iterator = maxhalonid - len(cells._halonid[i])
        for k in range(iterator):
             cells._halonid[i].append(-1)
        cells._halonid[i].append(numb)
        
    if size == 1 :
        cells._halonid = zeros((nbcells,2), dtype=int64)
        halos._halosext = zeros((2,2), dtype=int64)
        cells._loctoglob = zeros(nbcells)
        nodes._loctoglob = zeros(nbnodes)
        for i in range(nbnodes):
            nodes._loctoglob[i] = i
        for i in range(nbcells):
             cells._loctoglob[i] = i
    
    
    cells._halonid = asarray(cells._halonid, dtype=int64)
    nodes._halonid = asarray(nodes._halonid, dtype=int64)
    nodes._loctoglob = asarray(nodes._loctoglob, dtype=int64)
             
             
def create_3d_halo_structure(cells, faces, nodes, halos, size, nbcells, nbfaces, nbnodes):
   
    faces._halofid = zeros(nbfaces, dtype=int64)
    halos._nodes = zeros(nbnodes, dtype=int64)
    halos._faces = OrderedDict()
    
    facenameoldf = faces._name
    #TODO change adding cell index
    if size > 1:
        k = 1
        for i in range(len(halos._halosext)):
            halos._faces[tuple([halos._halosext[i][1], halos._halosext[i][2],  halos._halosext[i][3] ])] = k
            halos._faces[tuple([halos._halosext[i][3], halos._halosext[i][4],  halos._halosext[i][1] ])] = k + 1
            halos._faces[tuple([halos._halosext[i][1], halos._halosext[i][2],  halos._halosext[i][4] ])] = k + 2
            halos._faces[tuple([halos._halosext[i][4], halos._halosext[i][2],  halos._halosext[i][3] ])] = k + 3
            
            k = k+4

        for i in range(nbfaces):
            n1 = nodes._loctoglob[faces._nodeid[i][0]]
            n2 = nodes._loctoglob[faces._nodeid[i][1]]
            n3 = nodes._loctoglob[faces._nodeid[i][2]]
            
            if  halos._faces.get(tuple([n1, n2, n3])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                nodes._name[faces._nodeid[i][2]] = 10
               
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n1, n2, n3])))/4)
            
            if halos._faces.get(tuple([n1, n3, n2])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                nodes._name[faces._nodeid[i][2]] = 10
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n1, n3, n2])))/4)

            if halos._faces.get(tuple([n2, n1, n3])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                nodes._name[faces._nodeid[i][2]] = 10
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n2, n1, n3])))/4)
                
            if halos._faces.get(tuple([n2, n3, n1])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                nodes._name[faces._nodeid[i][2]] = 10
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n2, n3, n1])))/4)
                
            if halos._faces.get(tuple([n3, n1, n2])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                nodes._name[faces._nodeid[i][2]] = 10
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n3, n1, n2])))/4)
            
            if halos._faces.get(tuple([n3, n2, n1])):

                faces._cellid[i] = (faces._cellid[i][0], -10)
                faces._name[i] = 10
                nodes._name[faces._nodeid[i][0]] = 10
                nodes._name[faces._nodeid[i][1]] = 10
                nodes._name[faces._nodeid[i][2]] = 10
                faces._halofid[i] = int((-1+halos._faces.get(tuple([n3, n2, n1])))/4)
        
        #TODO verify halosext
        longueur = 0
        longh = []
        tmp = [[] for i in range(nbnodes)]
        for i in range(nbnodes):
            if nodes._name[i] == 10:
                halos._nodes[i] = nodes._loctoglob[i]
                arg = where(halos._halosext[:,1:5] == nodes._loctoglob[i])
                tmp[i].append(arg[0])
                longueur = max(longueur, len(arg[0]))
                longh.append(len(arg[0]))
            else:
                tmp[i].append(array([-1]))
                longh.append(0)

        nodes._halonid = [[-1]*longueur for i in range(nbnodes)]
        
        for i in range(len(tmp)):
            for j in range(len(tmp[i][0])):
                nodes._halonid[i][j] = tmp[i][0][j]
                
        for i in range(len(nodes._halonid)):
            nodes._halonid[i].append(longh[i])
                
    if size == 1 : 
        nodes._halonid = zeros((nbnodes,2), dtype=int64)
        halos._centvol  = zeros((2,2))
        halos._halosint = zeros((2,2))
        
    faces._ghostcenter = zeros((nbfaces, 4))
    nodes._ghostcenter = [[] for i in range(nbnodes)]
    nodes._ghostfaceinfo = [[] for i in range(nbnodes)]

    nodes._haloghostcenter = [[] for i in range(nbnodes)]
    nodes._haloghostfaceinfo = [[] for i in range(nbnodes)]

    kk = zeros(3)
    #TODO ghost center à verifier
    #compute the ghostcenter for each face and each node
    for i in range(nbfaces):
        if faces._name[i] != 0 and faces._name[i] < 7:
            
            nod1 = faces._nodeid[i][1]
            nod2 = faces._nodeid[i][0]
            nod3 = faces._nodeid[i][2]
            
            n = faces._normal[i]/faces._mesure[i]
            
            c_left = faces._cellid[i][0]
            v_1 = cells._center[c_left]
            u = faces._center[i][:] - v_1[:]
            gamma = dot(u, n)
            
            kk[0] = v_1[0] + gamma*n[0]
            kk[1] = v_1[1] + gamma*n[1]
            kk[2] = v_1[2] + gamma*n[2]
            
            v_2 = array([2 * kk[0] + ( -1 * v_1[0]), 2 * kk[1] + ( -1 * v_1[1]), 2 * kk[2] + ( -1 * v_1[2])])
            
            faces._ghostcenter[i] = [v_2[0], v_2[1], v_2[2], gamma]
            nodes._ghostcenter[nod1].append([v_2[0], v_2[1], v_2[2], faces._cellid[i][0], facenameoldf[i], i])
            nodes._ghostcenter[nod2].append([v_2[0], v_2[1], v_2[2], faces._cellid[i][0], facenameoldf[i], i])
            nodes._ghostcenter[nod3].append([v_2[0], v_2[1], v_2[2], faces._cellid[i][0], facenameoldf[i], i])
            
            ll = [faces._center[i][0], faces._center[i][1], faces._center[i][2], faces._normal[i][0], faces._normal[i][1], faces._normal[i][2]]
            nodes._ghostfaceinfo[nod1].append(ll)
            nodes._ghostfaceinfo[nod2].append(ll)
            nodes._ghostfaceinfo[nod3].append(ll)
            
        else:
            faces._ghostcenter[i] = [0.,0.,0., -1.]
    
    #define halo cells neighbor by nodes
    maxhalonid = 0
    cells._halonid = [[] for i in range(nbcells)]
    for i in range(nbcells):
        for j in range(4):
            nod = cells._nodeid[i][j]
            k = nodes._halonid[nod][-1]
            cells._halonid[i].extend(nodes._halonid[nod][:k])
        cells._halonid[i] = list(set(cells._halonid[i]))
        maxhalonid = max(maxhalonid, len(cells._halonid[i]))
    
    for i in range(nbcells):
        numb = len(cells._halonid[i])
        iterator = maxhalonid - len(cells._halonid[i])
        for k in range(iterator):
             cells._halonid[i].append(-1)
        cells._halonid[i].append(numb)
        
        
    if size == 1 :
        cells._halonid = zeros((nbcells,2), dtype=int64)
        halos._halosext = zeros((2,2), dtype=int64)
        nodes._loctoglob = zeros(nbnodes, dtype=int64)
        cells._loctoglob = zeros(nbcells, dtype=int64)
        for i in range(nbnodes):
            nodes._loctoglob[i]= i
        for i in range(nbcells):
             cells._loctoglob[i] = i
            
    
    cells._halonid = asarray(cells._halonid, dtype=int64)
    nodes._halonid = asarray(nodes._halonid, dtype=int64)
    nodes._loctoglob = asarray(nodes._loctoglob, dtype=int64)

# import collections
# def update_pediodic_info_2d_bis(nodes, cells, faces, halos, nbnodes, nbcells, 
#                                 nbfaces, loctoglob, comm, globalsize,  maxcoordx, maxcoordy, maxcoordz):
    
    
#     ##########################################################################
#     #TODO periodic test
#     nodes._periodicid = [[] for i in range(nbnodes)]
#     cells._periodicnid = [[] for i in range(nbcells)]
#     cells._periodiccenters = [[] for i in range(nbcells)]
#     cells._periodicfid = zeros(nbcells)
#     cells._shift = zeros((globalsize, 3))
    
#     #  TODO Periodic boundary (left and right)
#     leftb = {}
#     rightb = {}
#     leftbn =  collections.OrderedDict()
#     rightbn =  collections.OrderedDict()
#     leftbcn = collections.OrderedDict()
#     rightbcn =  collections.OrderedDict()
    
#     for i in range(0, nbfaces, 1):
#         if faces._name[i] == 11:
#             kk = tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])
#             leftb[kk] = loctoglob[faces._cellid[i][0]]
#             for j in range(cells._cellnid[faces._cellid[i][0]][-1]):
#                 leftbn.setdefault(kk, []).append(loctoglob[cells._cellnid[faces._cellid[i][0]][j]])
#                 leftbcn.setdefault(kk, []).append(cells._center[cells._cellnid[faces._cellid[i][0]][j]])
            
#         if faces._name[i] == 22:
#             kk = tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])
#             rightb[kk] = loctoglob[faces._cellid[i][0]]
#             for j in range(cells._cellnid[faces._cellid[i][0]][-1]):
#                 rightbn.setdefault(kk, []).append(loctoglob[cells._cellnid[faces._cellid[i][0]][j]])
#                 rightbcn.setdefault(kk, []).append(cells._center[cells._cellnid[faces._cellid[i][0]][j]])

    
#     rightbG = comm.allgather(rightb)
#     leftG = comm.allgather(leftb)
#     rightbnG = comm.allgather(rightbn)
#     leftbnG = comm.allgather(leftbn)
#     rightbcnG = comm.allgather(rightbcn)
#     leftbcnG = comm.allgather(leftbcn)
    
    
#     rightbGlob = {}
#     leftGlob = {}
    
#     rightbnGlob = {}
#     leftbnGlob = {}
    
#     rightbcnGlob = {}
#     leftbcnGlob = {}
    
    
#     for i in rightbG:
#         rightbGlob.update(i)
        
#     for i in leftG:
#         leftGlob.update(i)
        
#     for i in rightbnG:
#         rightbnGlob.update(i)
        
#     for i in leftbnG:
#         leftbnGlob.update(i)
        
#     for i in rightbcnG:
#         rightbcnGlob.update(i)
        
#     for i in leftbcnG:
#         leftbcnGlob.update(i)
    
    
#     longper = zeros(len(cells._center), dtype=int64)
    
    
#     for i in range(nbfaces):
#         if faces._name[i] == 11:
#             kk = tuple([float32(faces._center[i][0] + maxcoordx ), float32(faces._center[i][1]), float32(faces._center[i][2])])
#             faces._cellid[i][1] = rightbGlob[kk]
#             cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]
#             cells._periodicnid[faces._cellid[i][0]] = rightbnGlob[kk]
#             cells._periodiccenters[faces._cellid[i][0]] = rightbcnGlob[kk]
#             for cell in cells._periodicnid[faces._cellid[i][0]]:
#                   cells._shift[int(cell)][0] = -1*maxcoordx
#                   longper[faces._cellid[i][0]] +=1
            
#         elif faces._name[i] == 22:
#             kk = tuple([float32(faces._center[i][0] - maxcoordx), float32(faces._center[i][1]), float32(faces._center[i][2])])
#             faces._cellid[i][1] = leftGlob[kk]
#             cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
#             cells._periodicnid[faces._cellid[i][0]] = leftbnGlob[kk]
#             cells._periodiccenters[faces._cellid[i][0]] = leftbcnGlob[kk]
            
#             for cell in cells._periodicnid[faces._cellid[i][0]]:
#                 cells._shift[int(cell)][0] = maxcoordx
#                 longper[faces._cellid[i][0]] +=1
    
#     leftb = {}
#     rightb = {}
    
#     for i in range(0, nbnodes, 1):
#         for j in range(nodes._cellid[i][-1]):
#             if nodes.vertex[i][3] == 11:
#                 kk = tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])
#                 leftb.setdefault(kk, []).append(loctoglob[nodes._cellid[i][j]])
#             if nodes.vertex[i][3] == 22:
#                 kk = tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])
#                 rightb.setdefault(kk, []).append(loctoglob[nodes._cellid[i][j]])
    
    
#     rightbGlob = {}
#     leftGlob = {}
    
#     rightbG = comm.allgather(rightb)
#     leftG = comm.allgather(leftb)
    
#     for i in rightbG:
#         rightbGlob.update(i)
        
#     for i in leftG:
#         leftGlob.update(i)
        
#     for i in range(nbnodes):
#         if nodes.vertex[i][3] == 11:
#             kk = tuple([float32(nodes._vertex[i][0]) + maxcoordx, float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])
#             nodes._periodicid[i].extend(rightbGlob[kk])
#             nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1.]]
            
#         if nodes.vertex[i][3] == 22:
#             kk = tuple([float32(nodes._vertex[i][0]) - maxcoordx, float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])
#             nodes._periodicid[i].extend(leftGlob[kk])
#             nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1.]]
    
#     # ########################################################################################################
    
#     #  TODO Periodic boundary (upper and bottom)
#     leftb = {}
#     rightb = {}
#     leftbn =  collections.OrderedDict()
#     rightbn =  collections.OrderedDict()
    
#     for i in range(0, nbfaces, 1):
#         if faces._name[i] == 33:
#             kk = tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])
#             leftb[kk] = loctoglob[faces._cellid[i][0]]
#             for j in range(cells._cellnid[faces._cellid[i][0]][-1]):
#                 leftbn.setdefault(kk, []).append(loctoglob[cells._cellnid[faces._cellid[i][0]][j]])
            
#         if faces._name[i] == 44:
#             kk = tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])
#             rightb[kk] = loctoglob[faces._cellid[i][0]]
#             for j in range(cells._cellnid[faces._cellid[i][0]][-1]):
#                 rightbn.setdefault(kk, []).append(loctoglob[cells._cellnid[faces._cellid[i][0]][j]])
    
    
#     rightbG = comm.allgather(rightb)
#     leftG = comm.allgather(leftb)
#     rightbnG = comm.allgather(rightbn)
#     leftbnG = comm.allgather(leftbn)
    
#     rightbGlob = {}
#     leftGlob = {}
    
#     rightbnGlob = {}
#     leftbnGlob = {}
    
#     for i in rightbG:
#         rightbGlob.update(i)
        
#     for i in leftG:
#         leftGlob.update(i)
        
#     for i in rightbnG:
#         rightbnGlob.update(i)
        
#     for i in leftbnG:
#         leftbnGlob.update(i)
    
    
#     longper = zeros(len(cells._center), dtype=int64)
    
    
#     for i in range(nbfaces):
#         if faces._name[i] == 33:
#             kk = tuple([float32(faces._center[i][0]), float32(faces._center[i][1]  - maxcoordy ), float32(faces._center[i][2])])
#             faces._cellid[i][1] = rightbGlob[kk]
#             cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]
#             cells._periodicnid[faces._cellid[i][0]] = rightbnGlob[kk]
#             for cell in cells._periodicnid[faces._cellid[i][0]]:
#                   cells._shift[int(cell)][1] = -1*maxcoordy
#                   longper[faces._cellid[i][0]] +=1
            
#         elif faces._name[i] == 44:
#             kk = tuple([float32(faces._center[i][0]), float32(faces._center[i][1]  + maxcoordy ), float32(faces._center[i][2])])
#             faces._cellid[i][1] = leftGlob[kk]
#             cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
#             cells._periodicnid[faces._cellid[i][0]] = leftbnGlob[kk]
#             for cell in cells._periodicnid[faces._cellid[i][0]]:
#                 cells._shift[int(cell)][1] = maxcoordy
#                 longper[faces._cellid[i][0]] +=1
    
#     leftb = {}
#     rightb = {}
    
#     for i in range(0, nbnodes, 1):
#         for j in range(nodes._cellid[i][-1]):
#             if nodes.vertex[i][3] == 33:
#                 kk = tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])
#                 leftb.setdefault(kk, []).append(loctoglob[nodes._cellid[i][j]])
#             if nodes.vertex[i][3] == 44:
#                 kk = tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])
#                 rightb.setdefault(kk, []).append(loctoglob[nodes._cellid[i][j]])
    
    
#     rightbGlob = {}
#     leftGlob = {}
    
#     rightbG = comm.allgather(rightb)
#     leftG = comm.allgather(leftb)
    
#     for i in rightbG:
#         rightbGlob.update(i)
        
#     for i in leftG:
#         leftGlob.update(i)
        
#     for i in range(nbnodes):
#         if nodes.vertex[i][3] == 33:
#             kk = tuple([float32(nodes._vertex[i][0]) , float32(nodes._vertex[i][1] - maxcoordy)  , float32(nodes._vertex[i][2])])
#             nodes._periodicid[i].extend(rightbGlob[kk])
#             nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1.]]
            
#         if nodes.vertex[i][3] == 44:
#             kk = tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1] + maxcoordy ) , float32(nodes._vertex[i][2])])
#             nodes._periodicid[i].extend(leftGlob[kk])
#             nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1.]]
#     ###########################################################################################    
#     maxperiodiccell = 0
#     for i in range(nbcells):
#         cells._periodicnid[i] = unique(cells._periodicnid[i])
#         maxperiodiccell = max(maxperiodiccell, len(cells._periodicnid[i]))
        
    
#     for i in range(nbcells):
#         iterator  = maxperiodiccell - len(cells._periodicnid[i])
#         for j in range(iterator):
#             cells._periodicnid[i] = append(cells._periodicnid[i], -1)

#         if cells._periodicfid[i] == 0:
#             cells._periodicfid[i] = -1
#     for i in range(nbcells):
#           cells._periodicnid[i] = append(cells._periodicnid[i], len(cells._periodicnid[i][cells._periodicnid[i] !=-1]))
    
#     maxperiodicnode = 0
#     for i in range(nbnodes):
#         nodes._periodicid[i] = unique(nodes._periodicid[i])
#         maxperiodicnode = max(maxperiodicnode, len(nodes._periodicid[i]))
    
#     for i in range(nbnodes):
#         iterator  = maxperiodicnode - len(nodes._periodicid[i])
#         for j in range(iterator):
#             nodes._periodicid[i] = append(nodes._periodicid[i], -1) 
    
#     for i in range(nbnodes):
#         nodes._periodicid[i] = append(nodes._periodicid[i], len(nodes._periodicid[i][nodes._periodicid[i] !=-1]))
            
#     cells._loctoglob = asarray(cells._loctoglob, dtype=int64)
#     cells._periodicnid = asarray(cells._periodicnid, dtype=int64)
#     cells._periodicfid = asarray(cells._periodicfid, dtype=int64)
        
#     nodes._periodicid = asarray(nodes._periodicid, dtype=int64)
    
    
def update_pediodic_info_2d(nodes, cells, faces, halos, nbnodes, nbcells, periodicinfaces, periodicoutfaces, periodicupperfaces, periodicbottomfaces,
                            periodicinnodes, periodicoutnodes, periodicuppernodes, periodicbottomnodes):
    
    
    ##########################################################################
    #TODO periodic test
    nodes._periodicid = [[] for i in range(nbnodes)]
    cells._periodicnid = [[] for i in range(nbcells)]
    cells._periodicfid = zeros(nbcells)
    cells._shift = zeros((nbcells, 3))
    
    #  TODO Periodic boundary (left and right)
    leftb = {}
    rightb = {}
    for i in periodicinfaces:
        leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    for i in periodicoutfaces:
        rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    
    shiftx = max(nodes._vertex[:,0])
    maxcoordx = max(faces._center[:,0])
    longper = zeros(len(cells._center), dtype=int64)
    
    for i in periodicinfaces:
        faces._cellid[i][1] = rightb[tuple([float32(faces._center[i][0] + maxcoordx ), float32(faces._center[i][1]), float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#rightb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][0] = -1*shiftx
    
    for i in periodicoutfaces:
        faces._cellid[i][1] = leftb[tuple([float32(faces._center[i][0] - maxcoordx), float32(faces._center[i][1]), float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][0] = shiftx
    
    leftb = {}
    rightb = {}
    
    for i in periodicinnodes:
        for j in range(nodes._cellid[i][-1]):
            leftb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    for i in periodicoutnodes:
        for j in range(nodes._cellid[i][-1]):
            rightb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    
    for i in periodicinnodes:
        nodes._periodicid[i].extend(rightb[tuple([float32(nodes._vertex[i][0]) + maxcoordx, float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1.]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1.]]
    for i in periodicoutnodes:    
        nodes._periodicid[i].extend(leftb[tuple([float32(nodes._vertex[i][0]) - maxcoordx, float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1.]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1.]]
    
    ########################################################################################################
    #  TODO Periodic boundary (bottom and upper)
    leftb = {}
    rightb = {}
    for i in periodicupperfaces:
        leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    for i in periodicbottomfaces:
        rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    
    shifty = max(nodes._vertex[:,1])
    maxcoordy = max(faces._center[:,1])
    
    for i in periodicupperfaces:
        faces._cellid[i][1] = rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]) - maxcoordy, float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#rightb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][1] = -1*shifty
    for i in periodicbottomfaces:        
        faces._cellid[i][1] = leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1])  + maxcoordy, float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][1] = shifty
                    
    leftb = {}
    rightb = {}
    
    for i in periodicuppernodes:
        for j in range(nodes._cellid[i][-1]):
            leftb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    for i in periodicbottomnodes:
        rightb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    
    for i in periodicuppernodes:
        nodes._periodicid[i].extend(rightb[tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) - maxcoordy, float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1, -1, -1, -1, -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1.]]
    for i in periodicbottomnodes:    
        nodes._periodicid[i].extend(leftb[tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) + maxcoordy , float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1, -1, -1, -1, -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1.]]
            
    ###########################################################################################    
    maxperiodiccell = 0
    for i in range(nbcells):
        cells._periodicnid[i] = unique(cells._periodicnid[i])
        maxperiodiccell = max(maxperiodiccell, len(cells._periodicnid[i]))
        
    
    for i in range(nbcells):
        iterator  = maxperiodiccell - len(cells._periodicnid[i])
        for j in range(iterator):
            cells._periodicnid[i] = append(cells._periodicnid[i], -1)

        if cells._periodicfid[i] == 0:
            cells._periodicfid[i] = -1
    for i in range(nbcells):
          cells._periodicnid[i] = append(cells._periodicnid[i], len(cells._periodicnid[i][cells._periodicnid[i] !=-1]))
    
    maxperiodicnode = 0
    for i in range(nbnodes):
        nodes._periodicid[i] = unique(nodes._periodicid[i])
        maxperiodicnode = max(maxperiodicnode, len(nodes._periodicid[i]))
    
    for i in range(nbnodes):
        iterator  = maxperiodicnode - len(nodes._periodicid[i])
        for j in range(iterator):
            nodes._periodicid[i] = append(nodes._periodicid[i], -1) 
    
    for i in range(nbnodes):
        nodes._periodicid[i] = append(nodes._periodicid[i], len(nodes._periodicid[i][nodes._periodicid[i] !=-1]))
            
    cells._loctoglob = asarray(cells._loctoglob, dtype=int64)
    cells._periodicnid = asarray(cells._periodicnid, dtype=int64)
    cells._periodicfid = asarray(cells._periodicfid, dtype=int64)
        
    nodes._periodicid = asarray(nodes._periodicid, dtype=int64)
    
    
def update_pediodic_info_3d(nodes, cells, faces, halos, nbnodes, nbcells, periodicinfaces, periodicoutfaces, periodicupperfaces, periodicbottomfaces,
                            periodicfrontfaces, periodicbackfaces, periodicinnodes, periodicoutnodes, periodicuppernodes, periodicbottomnodes, 
                            periodicfrontnodes, periodicbacknodes):
    
    ##########################################################################
    #TODO periodic test
    nodes._periodicid = [[] for i in range(nbnodes)]
    cells._periodicnid = [[] for i in range(nbcells)]
    cells._periodicfid = zeros(nbcells)
    cells._shift = zeros((nbcells, 3))
    
    #  TODO Periodic boundary (left and right)
    leftb = {}
    rightb = {}
    for i in periodicinfaces:
        leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    for i in periodicoutfaces:
        rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    
    shiftx = max(nodes._vertex[:,0])
    maxcoordx = max(faces._center[:,0])
    longper = zeros(len(cells._center), dtype=int64)
    
    for i in periodicinfaces:
        faces._cellid[i][1] = rightb[tuple([float32(faces._center[i][0] + maxcoordx ), float32(faces._center[i][1]), float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#rightb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][0] = -1*shiftx
   
    for i in periodicoutfaces:        
        faces._cellid[i][1] = leftb[tuple([float32(faces._center[i][0] - maxcoordx), float32(faces._center[i][1]), float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][0] = shiftx
    
    leftb = {}
    rightb = {}
    
    for i in periodicinnodes:
        for j in range(nodes._cellid[i][-1]):
            leftb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    for i in periodicoutnodes:
        for j in range(nodes._cellid[i][-1]):
            rightb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    
    for i in periodicinnodes:
        nodes._periodicid[i].extend(rightb[tuple([float32(nodes._vertex[i][0]) + maxcoordx, float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1., -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1., -1., -1.]]
    for i in periodicoutnodes:        
        nodes._periodicid[i].extend(leftb[tuple([float32(nodes._vertex[i][0]) - maxcoordx, float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1., -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1., -1., -1.]]

    
    ########################################################################################################
    #  TODO Periodic boundary (bottom and upper)
    leftb = {}
    rightb = {}
    for i in periodicupperfaces:
        leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    for i in periodicbottomfaces:
        rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    
    shifty = max(nodes._vertex[:,1])
    maxcoordy = max(faces._center[:,1])
    
    for i in periodicupperfaces:
        faces._cellid[i][1] = rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]) - maxcoordy, float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#rightb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][1] = -1*shifty
    for i in periodicbottomfaces:    
        faces._cellid[i][1] = leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1])  + maxcoordy, float32(faces._center[i][2])])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][1] = shifty
                    
    leftb = {}
    rightb = {}
    
    for i in periodicuppernodes:
        for j in range(nodes._cellid[i][-1]):
            leftb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    for i in periodicbottomnodes:
        for j in range(nodes._cellid[i][-1]):
            rightb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    
    for i in periodicuppernodes:
        nodes._periodicid[i].extend(rightb[tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) - maxcoordy, float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1., -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1., -1., -1.]]
    
    for i in periodicbottomnodes:         
        nodes._periodicid[i].extend(leftb[tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) + maxcoordy , float32(nodes._vertex[i][2])])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1., -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1., -1., -1.]]
            
            
   ########################################################################################################
    #  TODO Periodic boundary (front and back)
    leftb = {}
    rightb = {}

    for i in periodicfrontfaces:
        leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    for i in periodicbackfaces:
        rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]), float32(faces._center[i][2])])] = faces._cellid[i][0]
    
    shiftz = max(nodes._vertex[:,2])
    maxcoordz = max(faces._center[:,2])
    
    for i in periodicfrontfaces:
        faces._cellid[i][1] = rightb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]) , float32(faces._center[i][2])- maxcoordz])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#rightb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][2] = -1*shiftz
    for i in periodicbackfaces:        
        faces._cellid[i][1] = leftb[tuple([float32(faces._center[i][0]), float32(faces._center[i][1]) , float32(faces._center[i][2]) + maxcoordz])]
        cells._periodicfid[faces._cellid[i][0]] = faces._cellid[i][1]#leftb[float32(faces._center[i][1])]
        for j in range(cells._cellnid[faces._cellid[i][1]][-1]):
            cells._periodicnid[faces._cellid[i][0]].append(cells._cellnid[faces._cellid[i][1]][j])
            longper[faces._cellid[i][0]] +=1
        for cell in cells._periodicnid[faces._cellid[i][0]]:
            if i != -1:
                cells._shift[cell][2] = shiftz
                
    leftb = {}
    rightb = {}
    
    for i in periodicfrontnodes:
        for j in range(nodes._cellid[i][-1]):
            leftb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    for i in periodicbacknodes:
        for j in range(nodes._cellid[i][-1]):
            rightb.setdefault(tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])]), []).append(nodes._cellid[i][j])
    
    for i in periodicfrontnodes:
        nodes._periodicid[i].extend(rightb[tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2])- maxcoordz])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1., -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1., -1., -1.]]
    for i in periodicbacknodes:        
        nodes._periodicid[i].extend(leftb[tuple([float32(nodes._vertex[i][0]), float32(nodes._vertex[i][1]) , float32(nodes._vertex[i][2]) + maxcoordz ])])
        nodes._ghostcenter[i] = [[-1., -1., -1., -1., -1., -1]]
        nodes._ghostfaceinfo[i] = [[-1., -1., -1., -1., -1., -1.]]

    ###########################################################################################    
    maxperiodiccell = 0
    for i in range(nbcells):
        cells._periodicnid[i] = unique(cells._periodicnid[i])
        maxperiodiccell = max(maxperiodiccell, len(cells._periodicnid[i]))
        
    
    for i in range(nbcells):
        iterator  = maxperiodiccell - len(cells._periodicnid[i])
        for j in range(iterator):
            cells._periodicnid[i] = append(cells._periodicnid[i], -1)

        if cells._periodicfid[i] == 0:
            cells._periodicfid[i] = -1
    for i in range(nbcells):
          cells._periodicnid[i] = append(cells._periodicnid[i], len(cells._periodicnid[i][cells._periodicnid[i] !=-1]))
    
    maxperiodicnode = 0
    for i in range(nbnodes):
        nodes._periodicid[i] = unique(nodes._periodicid[i])
        maxperiodicnode = max(maxperiodicnode, len(nodes._periodicid[i]))
    
    for i in range(nbnodes):
        iterator  = maxperiodicnode - len(nodes._periodicid[i])
        for j in range(iterator):
            nodes._periodicid[i] = append(nodes._periodicid[i], -1) 
    
    for i in range(nbnodes):
        nodes._periodicid[i] = append(nodes._periodicid[i], len(nodes._periodicid[i][nodes._periodicid[i] !=-1]))
            
    cells._loctoglob = asarray(cells._loctoglob, dtype=int64)
    cells._periodicnid = asarray(cells._periodicnid, dtype=int64)
    cells._periodicfid = asarray(cells._periodicfid, dtype=int64)
        
    nodes._periodicid = asarray(nodes._periodicid, dtype=int64)
        
@njit(fastmath=True)
def face_gradient_info_3d(cellidf:'int[:,:]', nodeidf:'int[:,:]', centergf:'float[:,:,:]', namef:'int[:]', normalf:'float[:,:]', 
                          mesuref:'float[:]', centerc:'float[:,:]',  centerh:'float[:,:]', 
                          halofid:'int[:]', vertexn:'float[:,:]', airDiamond:'float[:]', param1:'float[:]', param2:'float[:]', 
                          param3:'float[:]', n1:'float[:,:]',  n2:'float[:,:]',  shift:'float[:,:]', dim:'int'):
    
    nbfaces = len(cellidf)
    
    v_1 = zeros(dim)
    v_2 = zeros(dim)
    s1 = zeros(3)
    s2 = zeros(3)
    s3 = zeros(3)
    s4 = zeros(3)
    s5 = zeros(3)
    s6 = zeros(3)
    s7 = zeros(3)
    
    for i in range(nbfaces):

        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        v_1[:] = centerc[c_left][0:dim]
        
        if namef[i] == 0:
            v_2[:] = centerc[c_right][0:dim]
        
        elif namef[i] == 11 or namef[i] == 22 :
            v_2[0] = centerc[c_right][0] + shift[c_right][0]
            v_2[1] = centerc[c_right][1] 
            v_2[2] = centerc[c_right][2]
        elif namef[i] == 33 or namef[i] == 44:
            v_2[0] = centerc[c_right][0]
            v_2[1] = centerc[c_right][1] + shift[c_right][1]
            v_2[2] = centerc[c_right][2]
        elif namef[i] == 55 or namef[i] == 66:
            v_2[0] = centerc[c_right][0]
            v_2[1] = centerc[c_right][1] 
            v_2[2] = centerc[c_right][2] + shift[c_right][2]
            
        elif namef[i] == 10:
            v_2[:] = centerh[halofid[i]][0:dim]
        else :
            v_2[:] = centergf[i][0:dim]
        
        # if namef[i] == 0:
        #     v_2[:] = centerc[c_right][0:dim]
        # elif namef[i] == 10:
        #     v_2[:] = centerh[halofid[i]][0:dim]
        # else :
        #     v_2[:] = centerg[i][0:dim]

        s1[:] = v_2                 - vertexn[i_2][0:dim]
        s2[:] = vertexn[i_4][0:dim] - vertexn[i_2][0:dim]
        s3[:] = v_1                 - vertexn[i_2][0:dim]
        n1[i][:] = (0.5 * cross(s1, s2)) + (0.5 * cross(s2, s3))
        
        s4[:] = v_2                 - vertexn[i_3][0:dim]
        s5[:] = vertexn[i_1][0:dim] - vertexn[i_3][0:dim]
        s6[:] = v_1                 - vertexn[i_3][0:dim]
        n2[i][:] = (0.5 * cross(s4, s5)) + (0.5 * cross(s5, s6))
        
        s7[:] = v_2 - v_1
        airDiamond[i] = dot(normalf[i], s7)
        
        param1[i] = (dot(n1[i], normalf[i])) / airDiamond[i]
        param2[i] = (dot(n2[i], normalf[i])) / airDiamond[i]
        param3[i] = (dot(normalf[i], normalf[i])) / airDiamond[i]