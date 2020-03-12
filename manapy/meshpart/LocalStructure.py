#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:10:58 2020

@author: kissami
"""
from mpi4py import MPI
from collections import OrderedDict 
import numpy as np
import meshio
from numba import njit, literal_unroll
from numba.typed import  Dict


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ["generate_mesh"]



class Cells:
    nodeid = []
    faceid = []
    center = []
    volume = []
    globalIndex = OrderedDict()
    father = Dict()
    son = Dict()
    Iadiv = Dict()
    
    def __init__(self, nodeid, faceid, center, volume, father, son, Iadiv, globalIndex):
        self.nodeid = nodeid    # instance variable unique to each instance
        self.faceid = faceid
        self.center = center
        self.volume = volume
        self.father = father
        self.son = son
        self.Iadiv = Iadiv
        self.globalIndex = globalIndex
    

class Nodes:
    vertex = []
    name = []
    #pas encore créé
    cellid = []
    globalIndex = OrderedDict()
    
    def __init__(self, vertex, name, cellid, globalIndex):
        self.vertex = vertex    # instance variable unique to each instance
        self.name = name
        self.cellid = cellid
        self.globalIndex = globalIndex
        
class Faces:
    nodeid = []
    cellid = []
    name = []
    normal = []
    bound = 0
    
    def __init__(self, nodeid, cellid, name, normal, bound):
        self.nodeid = nodeid    # instance variable unique to each instance
        self.cellid = cellid
        self.name   = name
        self.normal = normal
        self.bound  = bound 
    
class Halo:
    halosInt = []
    halosExt = []
    neigh = []
    faces = OrderedDict()
   
    def __init__(self, halosInt, halosExt, neigh, faces):
        self.halosInt = halosInt    # instance variable unique to each instance
        self.halosExt = halosExt
        self.neigh = neigh
        self.faces = faces
@njit
def VecteurNormal(a,b,bary):

    n = np.zeros(2)#[None]*2
    s = np.zeros(2)#[None]*2
    m = np.zeros(2)
    normal = np.zeros(2)#[None]*2
     
   
    n[0] = a[1] - b[1]
    n[1] = b[0] - a[0];
    
    m[0] = 0.5 * (a[0] + b[0]);
    m[1] = 0.5 * (a[1] + b[1]);
    
    s[0] = bary[0] - m[0] ;
    s[1] = bary[1] - m[1] ;
    
    if ( (s[0] * n[0] + s[1] * n[1]) > 0):
        normal[0] = -1*n[0];
        normal[1] = -1*n[1];
    else:
        normal[0] = n[0];
        normal[1] = n[1];
        
    normal[0] = normal[0]#/longueur
    normal[1] = normal[1]#/longueur 
    
    
    return normal#, longueur
 
def create_local_mesh(file):
    
   
    clear_class(Cells, Nodes, Faces)    
    #Lecture des cellules à partir du fichier mesh..txt
    #file = open("mesh"+str(rank)+".txt","r")
    for line in file:
        #read elements
        if line == "Elements\n":
            continue   
        if (line == "EndElements\n"):
            continue
        if line == "Nodes\n":
            break
        Cells.nodeid.append( [int(x) for x in line.split()] )
    #Lecture des coordonnées des noeuds à partir du fichier mesh..txt
    for line in file:
        #read Nodes
        if line == "Nodes\n":
            continue   
        if (line == "EndNodes\n"):
            continue
        if line == "HalosInt\n":
            break
        Nodes.vertex.append([float(x) for x in line.split()])
        Nodes.name = [0]*len(Nodes.vertex)
    
    for i in range(len(Nodes.vertex)):
            Nodes.name[i] = int(Nodes.vertex[i][3])
        
    #calcul du barycentre      
    for i in range(len(Cells.nodeid)):
        s1 = Cells.nodeid[i][0]
        s2 = Cells.nodeid[i][1]
        s3 = Cells.nodeid[i][2]
        
        x1 = Nodes.vertex[s1][0]
        y1 = Nodes.vertex[s1][1]
        x2 = Nodes.vertex[s2][0]
        y2 = Nodes.vertex[s2][1]
        x3 = Nodes.vertex[s3][0]
        y3 = Nodes.vertex[s3][1]
        
            
        Cells.center.append((1./3 * (x1 + x2 + x3), 1./3*(y1 + y2 + y3)))
        
        Cells.volume.append( (1./2) * abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2)))

    

    #Création des faces  
    Cellule = Cells.nodeid
    cellF = []
    faces = []
    k = 0
    for i in range(len(Cellule)):
        faces.append([Cellule[i][0], Cellule[i][1]])
        faces.append([Cellule[i][1], Cellule[i][2]])
        faces.append([Cellule[i][0], Cellule[i][2]])
        cellF.append([faces[k], faces[k+1], faces[k+2]])
        k = k+3
    
    faces =   set( tuple(x) for x in faces)
    faces = list(faces)
    
    facesdict = OrderedDict()
    for i in range(len(faces)):
        facesdict[faces[i]] = i
        Faces.nodeid.append(faces[i])
        Faces.cellid.append((-1,-1))
 
    
        
    #Création des 3 faces de chaque cellule            
    for i in range(len(cellF)):
        Cells.faceid.append( [facesdict.get(tuple(cellF[i][0])), facesdict.get(tuple(cellF[i][1])), 
                            facesdict.get(tuple(cellF[i][2]))])
        
    #Faces.cellid = [(-1, -1)]*len(faces)
    for i in range(len(Cells.faceid)):
        for j in range(3):
            if Faces.cellid[Cells.faceid[i][j]] == (-1, -1):
                Faces.cellid[Cells.faceid[i][j]]= (i, -1)
            if Faces.cellid[Cells.faceid[i][j]][0] != i:
                Faces.cellid[Cells.faceid[i][j]]= (Faces.cellid[Cells.faceid[i][j]][0], i) 
    
   
    
 
    #Faces aux bords (1,2,3,4), Faces à l'interieur 0    A VOIR !!!!!   
    for i in range(len(faces)):
        Faces.name.append(0)
        if (Faces.cellid[i][1] == -1 and Faces.cellid[i][1] != -10):
            if (Nodes.name[faces[i][0]] == Nodes.name[faces[i][1]] ):
                Faces.name[i] = Nodes.name[faces[i][0]]
            if ((Nodes.name[faces[i][0]] == 3 and Nodes.name[faces[i][1]] !=0) or 
                (Nodes.name[faces[i][0]] != 0 and Nodes.name[faces[i][1]] ==3)):
                Faces.name[i] = 3
            if ((Nodes.name[faces[i][0]] == 4 and Nodes.name[faces[i][1]] !=0) or 
                (Nodes.name[faces[i][0]] != 0 and Nodes.name[faces[i][1]] ==4)):
                Faces.name[i] = 4
        Faces.normal.append( VecteurNormal(np.asarray(Nodes.vertex[faces[i][0]]),
                                           np.asarray(Nodes.vertex[faces[i][1]]), 
                                           Cells.center[Faces.cellid[i][0]]))
    for i in range(len(faces)):
        if Faces.name[i] != 0:
            Faces.bound += 1


def create_halo_structure(file):
    
     
    #txt_file = open(filename)
    for line in file:
        if "EndHalosInt" in line:
            break
        Halo.halosInt.append(int(line))#.append(int(line))#for x in line.split()])
    for line in file:
        # process the line
        if "HalosExt" in line :
            continue
        if "GlobalCellToLocal" in line:
            break
        Halo.halosExt.append( [int(x) for x in line.split()])
    
    #Cells.globalIndex = [-1]*len(Cells.nodeid)
    cmpt = 0
    for line in file:
    #read Global cell to local
        if line == "GlobalCellToLocal\n":
            continue   
        if (line == "EndGlobalCellToLocal\n"):
            break
        Cells.globalIndex[int(line)] = cmpt
        cmpt +=1
    
    cmpt = 0
    for line in file:
        #read Local Node To Global
        if line == "LocalNodeToGlobal\n":
            continue   
        if (line == "EndLocalNodeToGlobal\n"):
            break
        Nodes.globalIndex[cmpt] = int(line)
        cmpt +=1
        #Nodes.GlobalIndex.append(int(line))# for x in line.split()])  
    for line in file:
        #read LocalToGlobal
        if line == "Neigh\n":
            continue   
        if (line == "EndNeigh\n"):
            break
        Halo.neigh.append([int(x) for x in line.split()])#int(line))
               
    #halofaces = []
    if size > 1:
        k = 1
        for i in range(len(Halo.halosExt)):
            Halo.faces[tuple([Halo.halosExt[i][0], Halo.halosExt[i][1]])] = k
            Halo.faces[tuple([Halo.halosExt[i][1], Halo.halosExt[i][2]])] = k + 1
            Halo.faces[tuple([Halo.halosExt[i][0], Halo.halosExt[i][2]])] = k + 2
            k = k+3
    
        for i in range(len(Faces.nodeid)):
            
            if  Halo.faces.get(tuple([Nodes.globalIndex[Faces.nodeid[i][0]],
                                      Nodes.globalIndex[Faces.nodeid[i][1]]])):
               # print(tuple([Nodes.globalIndex[Faces.nodeid[i][0]],
                #                      Nodes.globalIndex[Faces.nodeid[i][1]]]))
                Faces.cellid[i] = (Faces.cellid[i][0], -10)
                Faces.name[i] = 10
                #print(Faces.name[i], i, rank)
        
    cells = Cells(np.asarray(Cells.nodeid), np.asarray(Cells.faceid), np.asarray(Cells.center), 
                  np.asarray(Cells.volume), Cells.father, 
                  Cells.son, Cells.Iadiv, Cells.globalIndex)
    nodes = Nodes(np.asarray(Nodes.vertex), np.asarray(Nodes.name), np.asarray(Nodes.cellid), Nodes.globalIndex)
    faces = Faces(np.asarray(Faces.nodeid), np.asarray(Faces.cellid), np.asarray(Faces.name), 
                  np.asarray(Faces.normal), Faces.bound)
    halos = Halo(np.asarray(Halo.halosInt), np.asarray(Halo.halosExt), np.asarray(Halo.neigh),
                 Halo.faces)
    
   
    return cells, nodes, faces, halos


@njit
def ghost_value(w, w_ghost, cellid, name):
    
    for i in range(len(cellid)):
        if (name[i] == 1 or name[i] == 3 or name[i] ==4 ):
            w_ghost[i] = w[cellid[i][0]]
           # w_ghost[i].hu = 0
            #w_ghost[i].hv = 0

        elif name[i] !=0:
             w_ghost[i] = w[cellid[i][0]]
    return w_ghost

@njit
def define_halosend(w, w_halosend, cellid, indsend, mystruct):

  
    for i in range(len(w_halosend)):
        w_halosend[i] = w[indsend[i]]
      
    return w_halosend
            
        
def all_to_all(w_halosend, taille, mystruct, variables, scount, sdepl, rcount, rdepl):
    
    w_halorecv = np.zeros(taille, dtype = mystruct )
    
    for var in literal_unroll(variables):
        s_msg = r_msg = 0
        s_msg = [np.array(w_halosend[var]), (scount, sdepl), MPI.DOUBLE_PRECISION]
        r_msg = [np.array(w_halorecv[var]), (rcount, rdepl), MPI.DOUBLE_PRECISION]
           
        comm.Alltoallv(s_msg, r_msg)

        w_halorecv[var] = r_msg[0]        
   
    return w_halorecv

@njit
def halo_value(w_halo, cellid, w_halorecv, indRecv, mystruct):

    for i in range(len(w_halo)):
            if cellid[i][1] == -10:
                w_halo[i] = w_halorecv[indRecv[i]]


    return w_halo

def prepare_comm(cells, faces, nodes, halos):    
    
    scount = np.zeros(size, dtype = int)
    sdepl  = np.zeros(size, dtype = int)
    rcount = np.zeros(size, dtype = int)
    rdepl  = np.zeros(size, dtype = int)
        
    
    for i in range(len(halos.neigh[0])):
        scount[halos.neigh[0][i]] = halos.neigh[1][i]
    
    for i in range(size):
        if (i>0):
            sdepl[i] = sdepl[i-1] + scount[i-1]
    
    rcount = comm.alltoall(scount)
    
 #   print(rcount, scount , rank)
    for i in range(size):
        if (i>0):
            rdepl[i] = rdepl[i-1] + rcount[i-1]
  #  print(rcount,rdepl, scount , sdepl, rank)

    taille = 0
    for i in range(size):
        taille += rcount[i]
    
    taille = int(taille)
    
    indSend = np.zeros(0, dtype = int)
    for i in range(len(halos.halosInt)):
            indSend = np.append(indSend, cells.globalIndex[Halo.halosInt[i]])
    
    indRecv = np.zeros(len(faces.cellid), dtype = int)
    for i in range(len(faces.cellid)):
        if faces.cellid[i][1] == -10: 
            indRecv[i] = int((-1+halos.faces.get(
                    tuple([nodes.globalIndex[Faces.nodeid[i][0]], 
                           nodes.globalIndex[Faces.nodeid[i][1]]])))/3)

    return scount, sdepl, rcount, rdepl, taille, indSend, indRecv

def generate_mesh():

    filename = 'mesh'+str(rank)+'.txt'
    txt_file = open(filename)

    create_local_mesh(txt_file)
         
    
    cells, nodes, faces, halos = create_halo_structure(txt_file)

    Grid = {}

    Grid["cells"] = cells
    Grid["nodes"] = nodes
    Grid["faces"] = faces
    Grid["halos"] = halos
    
    txt_file.close()
    

    return Grid#cells, nodes, faces, halos

def clear_class(cells, nodes, faces):
    
    cells.nodeid = [] 
    cells.volume = []#OrderedDict()
    cells.center = []
    cells.faceid = []
    cells.father = Dict()
    cells.son    = Dict()
    cells.Iadiv  = Dict()
    cells.globalIndex = OrderedDict()
    
    faces.nodeid = []
    faces.name = []
    faces.cellid = []
    faces.normal = []
    faces.bound  = 0
    
    nodes.cellid = []
    nodes.name = []
    nodes.vertex = []
    nodes.globalIndex = OrderedDict()
    
@njit    
def compute_flux_advection(flux, Fl, Fr, wl, wr, n, mystruct):

    c = 0
    u = np.zeros(2)
    
    u[0] = 0.5*(wl.hu + wr.hu)/wr.h
    u[1] = 0.5*(wl.hv + wr.hv)/wr.h
 
    q = np.dot(u,n) 
    
    if (q >= 0):
        c = wl.h
    else:
        c = wr.h
   
    flux.h = q*c 
    flux.hu = 0
    flux.hv = 0
    
    return flux

@njit 
def compute_flux_shallow_roe(flux, Fl, Fr, wl, wr, n, mystruct):
    grav = 9.81
    
    WL = wl
    WR = wr
    
    #print(WL, WR)
    
    t = np.zeros(2)
    t[0] = -1*n[1]
    t[1] = n[0]
    mest = np.sqrt(t[0]*t[0] + t[1]*t[1])
    mesn = np.sqrt(n[0]*n[0] + n[1]*n[1])
     

    ul = WL.hu*n[0] + WL.hv*n[1]
    ul = ul / mesn
    vl = WL.hu*t[0] + WL.hv*t[1]
    vl = vl / mest
    #WL.hu = ul
    #WL.hv = vl
    
    ur = WR.hu*n[0] + WR.hv*n[1]
    ur = ur / mesn
    vr = WR.hu*t[0] + WR.hv*t[1]
    vr = vr / mest
    #WR.hu = ur
    #WR.hv = vr


    U_LRh =  (WL.h  + WR.h)/2
    U_LRhu = (ul/WL.h + vl/WR.h)/2
    U_LRhv = (ur/WL.h + vr/WR.h)/2
    
        
    c = np.sqrt(grav * U_LRh)
    
    lambda1 = U_LRhu - c
    lambda2 = U_LRhu
    lambda3 = U_LRhu + c;

        
    R = np.zeros((3,3))
    RI = np.zeros((3,3))
    AL = np.zeros((3,3))
    RAL = np.zeros((3,3))
    AM = np.zeros((3,3))

    AL[0][0] = np.fabs(lambda1)
    AL[1][0] = 0.
    AL[2][0] = 0.
    AL[0][1] = 0.
    AL[1][1] = np.fabs(lambda2)
    AL[2][1] = 0.
    AL[0][2] = 0.
    AL[1][2] = 0.
    AL[2][2] = np.fabs(lambda3)

    R[0][0] = 1.
    R[1][0] = lambda1
    R[2][0] = U_LRhv
    R[0][1] = 0.
    R[1][1] = 0.
    R[2][1] = 1.
    R[0][2] = 1.
    R[1][2] = lambda3
    R[2][2] = U_LRhv

    RI[0][0] = lambda3/(2*c)
    RI[1][0] = -U_LRhv
    RI[2][0] = -lambda1/(2*c)
    RI[0][1] = -1./(2*c)
    RI[1][1] = 0.
    RI[2][1] = 1./(2*c)
    RI[0][2] = 0.
    RI[1][2] = 1.
    RI[2][2] = 0.
       
    RAL = matmul(R, AL)
    AM  = matmul(RAL, RI)
    
    

    W_dif= np.zeros(3)
    W_dif[0] = WR.h  - WL.h
    W_dif[1] = ur    - ul #WR.hu - WL.hu
    W_dif[2] = vr    - vl #WR.hv - WL.hv
    
    h = 0.
    u = 0.
    v = 0.

    for i in range(3):
       h += AM[0][i] * W_dif[i]
       u += AM[1][i] * W_dif[i]
       v += AM[2][i] * W_dif[i]
        
    U_h = h/2
    U_hu = u/2
    U_hv = v/2
   
    ql = ul#WL.hu
    qr = ur#WR.hu
    
    pl = 0.5 * grav * WL.h*WL.h
    pr = 0.5 * grav * WR.h*WR.h
    
    Fl.h  = ql
    Fl.hu = ql * ul/WL.h + pl #WL.hu/WL.h +pl
    Fl.hv = ql * vl/WL.h      #WL.hv/WL.h
    
      
    Fr.h  = qr
    Fr.hu = qr * ur/WL.h + pr #WR.hu/WR.h +pr
    Fr.hv = qr * vr/WL.h      #WR.hv/WR.h
    
    
    FH  = 0.5 * (Fl.h + Fr.h) - U_h
    FHU = 0.5 * (Fl.hu + Fr.hu) - U_hu
    FHV = 0.5 * (Fl.hv + Fr.hv) - U_hv
    
    
    
    flux.h  = FH * mesn 
    flux.hu = (FHU*n[0] + FHV*-1*n[1])
    flux.hv = (FHU*-1*t[0] + FHV*t[1])
    
    
    return flux

@njit 
def compute_flux_shallow_srnh(flux, Fl, Fr, wl, wr, n, mystruct):
    grav = 9.81
    
    WL = wl
    WR = wr

    t = np.zeros(2)
    t[0] = -1*n[1]
    t[1] = n[0]
    mest = np.sqrt(t[0]*t[0] + t[1]*t[1])
    mesn = np.sqrt(n[0]*n[0] + n[1]*n[1])
   
    
    uh = (WL.hu / WL.h * np.sqrt(WL.h) + WR.hu / WR.h * np.sqrt(WR.h)) /(np.sqrt(WR.h) + np.sqrt(WR.h))
    vh = (WL.hv / WL.h * np.sqrt(WL.h) + WR.hv / WR.h * np.sqrt(WR.h)) /(np.sqrt(WR.h) + np.sqrt(WR.h))
    
        
    
    #uvh = np.array([uh, vh])
    Uh = uh*n[0] + vh*n[1]#np.dot(uvh , n);
    Uh = Uh / mesn
    Vh = uh*t[0] + vh*t[1]#np.dot(uvh , t);
    Vh = Vh / mest
    
    
        
    HROE = (WL.h+WR.h)/2
    UROE = Uh
    VROE = Vh
    
    ul = WL.hu*n[0] + WL.hv*n[1]
    ul = ul / mesn
    vl = WL.hu*t[0] + WL.hv*t[1]
    vl = vl / mest
    #WL.hu = ul
    #WL.hv = vl
    
    ur = WR.hu*n[0] + WR.hv*n[1]
    ur = ur / mesn
    vr = WR.hu*t[0] + WR.hv*t[1]
    vr = vr / mest
    #WR.hu = ur
    #WR.hv = vr


    W_LRh =  (WL.h  + WR.h)/2
    W_LRhu = (ul + ur)/2
    W_LRhv = (vl + vr)/2
    
        
    c = np.sqrt(grav * HROE)
    
    lambda1 = UROE - c
    lambda2 = UROE
    lambda3 = UROE + c;

    if (lambda1 == 0):
        sign1 = 0.;
    else:
        sign1 = lambda1 / np.fabs(lambda1);

    if (lambda2 == 0):
        sign2 = 0.;
    else:
        sign2 = lambda2 / np.fabs(lambda2);

    if (lambda3 == 0):
        sign3 = 0.
    else:
        sign3 = lambda3 / np.fabs(lambda3);
        
    R = np.zeros((3,3))
    RI = np.zeros((3,3))
    SL = np.zeros((3,3))
    RSL = np.zeros((3,3))
    SM = np.zeros((3,3))

    SL[0][0] = sign1
    SL[1][0] = 0.
    SL[2][0] = 0.
    SL[0][1] = 0.
    SL[1][1] = sign2
    SL[2][1] = 0.
    SL[0][2] = 0.
    SL[1][2] = 0.
    SL[2][2] = sign3

    R[0][0] = 1.
    R[1][0] = lambda1
    R[2][0] = VROE
    R[0][1] = 0.
    R[1][1] = 0.
    R[2][1] = 1.
    R[0][2] = 1.
    R[1][2] = lambda3
    R[2][2] = VROE

    RI[0][0] = lambda3/(2*c)
    RI[1][0] = -VROE
    RI[2][0] = -lambda1/(2*c)
    RI[0][1] = -1./(2*c)
    RI[1][1] = 0.
    RI[2][1] = 1./(2*c)
    RI[0][2] = 0.
    RI[1][2] = 1.
    RI[2][2] = 0.
       
    RSL = matmul(R, SL)
    SM  = matmul(RSL, RI)
    
   

    W_dif= np.zeros(3)
    W_dif[0] = WR.h  - WL.h
    W_dif[1] = ur - ul
    W_dif[2] = vr - vl
    
    h = 0.
    u = 0.
    v = 0.

    for i in range(3):
       h += SM[0][i] * W_dif[i]
       u += SM[1][i] * W_dif[i]
       v += SM[2][i] * W_dif[i]
    

    
    U_h = h/2
    U_hu = u/2
    U_hv = v/2
   
        
    W_LRh  = W_LRh - U_h
    W_LRhu = W_LRhu - U_hu
    W_LRhv = W_LRhv - U_hv
    
    
    u = 0.
    v = 0. 
    
    u = W_LRhu * n[0] + W_LRhv * -1*n[1]
    u = u / mesn
    v = W_LRhu * -1*t[0] + W_LRhv * t[1]
    v = v / mest
    
    
    W_LRhu = u
    W_LRhv = v

    q = n[0] * u + n[1] * v
    
     
    
    flux.h = q
    flux.hu = q * W_LRhu/W_LRh + 0.5 * grav * W_LRh * W_LRh * n[0]
    flux.hv = q * W_LRhv/W_LRh + 0.5 * grav * W_LRh * W_LRh * n[1]

    return flux

@njit
def compute_flux_shallow_rusanov(flux, Fl, Fr, wl, wr, n, mystruct):
    
    grav = 9.81
    
    ql = wl.hu * n[0] + wl.hv * n[1]
    pl = 0.5 * grav * wl.h * wl.h
    
    mes = np.sqrt(n[0]*n[0] + n[1]*n[1])
    
    qr = wr.hu * n[0] + wr.hv * n[1]
    pr = 0.5 * grav * wr.h * wr.h
    


    Fl.h  = ql
    Fl.hu  = ql * wl.hu/wl.h + pl*n[0]
    Fl.hv  = ql * wl.hv/wl.h + pl*n[1]
    
    
    Fr.h  = qr
    Fr.hu  = qr * wr.hu/wr.h + pr*n[0]
    Fr.hv  = qr * wr.hv/wr.h + pr*n[1]

    cl = np.sqrt(grav * wl.h);
    cr = np.sqrt(grav * wr.h);
  
    Ll = np.fabs((ql/mes)/wl.h) + cl
    Lr = np.fabs((qr/mes)/wr.h) + cr

    if (Ll > Lr): 
        S = Ll;
    else: 
        S = Lr;
    
    flux.h = 0.5 * (Fl.h + Fr.h) - 0.5 * S * mes * (wr.h - wl.h)
    flux.hu = 0.5 * (Fl.hu + Fr.hu) - 0.5 * S * mes * (wr.hu - wl.hu)
    flux.hv = 0.5 * (Fl.hv + Fr.hv) - 0.5 * S * mes * (wr.hv - wl.hv)
#    

    return flux

@njit
def add(a,b):
    a.h += b.h
    a.hu += b.hu
    a.hv += b.hv
    return a
@njit
def minus(a,b):
    a.h -= b.h
    a.hu -= b.hu
    a.hv -= b.hv
    return a

@njit
def matmul(matrix1,matrix2):
    rmatrix = np.zeros((3,3))
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix
@njit
def ExplicitScheme(w, w_ghost, w_halo, cellid, normal, name, mystruct, variables):
    
   
    rezidus  = np.zeros(len(w), dtype = mystruct )#np.zeros(len(w))  
    wl = np.zeros(1, dtype = mystruct )[0]
    wr = np.zeros(1, dtype = mystruct )[0]
    flx = np.zeros(1, dtype = mystruct )[0]
    Fl  = np.zeros(1, dtype = mystruct )[0]
    Fr  = np.zeros(1, dtype = mystruct )[0]
    
   
    for i in range(len(cellid)):
        wl = w[cellid[i][0]]  
        n = normal[i]

        if (name[i] == 0):
            wr = w[cellid[i][1]]
            
            flx = compute_flux_shallow_srnh(flx, Fl, Fr, wl, wr, n, mystruct) 

            rezidus[cellid[i][0]] =  minus(rezidus[cellid[i][0]],flx)
            rezidus[cellid[i][1]] =  add(rezidus[cellid[i][1]],flx)
                        
        elif (name[i] == 10):
            wr = w_halo[i]

            flx = compute_flux_shallow_srnh(flx, Fl, Fr, wl, wr, n, mystruct)
            rezidus[cellid[i][0]] =  minus(rezidus[cellid[i][0]],flx)
                             #        
        else:
            wr = w_ghost[i]
            flx = compute_flux_shallow_srnh(flx, Fl, Fr, wl, wr,  n, mystruct)
            rezidus[cellid[i][0]] =  minus(rezidus[cellid[i][0]],flx)
  
    return rezidus   

@njit
def update(w, wn, dt, rezidus, volume):
     for i in range(len(w)):
        wn.h[i]= w.h[i] + dt * (rezidus["h"][i]/volume[i])
        wn.hu[i]= w.hu[i] + dt * (rezidus["hu"][i]/volume[i])
        wn.hv[i]= w.hv[i] + dt * (rezidus["hv"][i]/volume[i])
     return wn

@njit
def time_step(w, cfl, normal, volume, faceid):
       
    dt_c = np.zeros(len(faceid))
   
    for i in range(len(faceid)):
        a = np.sqrt(9.81*w[i].h)
        lam = 0
        for j in range(3):
            s = normal[faceid[i][j]] 
            u_n = np.fabs(w.hu[i]/w.h[i]*s[0] + w.hv[i]/w.h[i]*s[1])#np.dot(w.hu[i],s))
            mes = np.sqrt(s[0]*s[0] + s[1]*s[1])
            lam_j = u_n/mes + a
            lam += lam_j * mes
           
        dt_c[i] = cfl * volume[i]/lam

    dt = np.asarray(np.min(dt_c))
            
    return dt

@njit
def initialisation(w, center):
    
   
    nbelements = len(center)
    sigma = 2;

    choix = 1 # (0,creneau 1:gaussienne)
    if choix == 0:
        for i in range(nbelements):
            x = center[i][0]
            y = center[i][1]
            w.h[i] = 5 * np.exp(-1.*(pow(x-0, 2) + pow(y-0, 2)) / (pow(sigma, 2)))
            w.hu[i] = 0.
            w.hv[i] = 0.
    
    elif choix == 1:
        for i in range(nbelements):
            if(center[i][0] < 5):
                w.h[i] = 5.
            else:
                w.h[i] = 2.
            w.hu[i] = 0.
            w.hv[i] = 0.
 
    elif choix == 2:
        for i in range(nbelements):
            x = center[i][0]
            y = center[i][1]
            a = 2.5; b = 0.4; c= 1;
            
    
            w.h[i]  = 1. + 0.25*(1-np.tanh(c*(np.sqrt(a*x**2 + b*y**2)-1.)));
            w.hu[i] = 0
            w.hv[i] = 0
   
    elif choix == 3:
        a = 0.04 ; b = 0.02 ; g = 9.81
        xbar = ybar = [0]*nbelements
        theta = np.pi/6
        for i in range(nbelements):
            
            x = center[i][0]
            y = center[i][1]
            xbar[i] = x# + 20 #-0.5*t*cos(theta);
            ybar[i] = y# + 10 #-0.5*t*sin(theta);
    
            w.h[i]  = 1 - a**2/(4*b*g) * np.exp(-2*b*(pow(xbar[i],2) + pow(ybar[i],2)))#ybar.^2));            
            w.hu[i] = 0.5 * np.cos(theta) + a*ybar[i] * np.exp(-b*(pow(xbar[i],2) + pow(ybar[i],2)));
            w.hv[i] = 0.5 * np.sin(theta) - a*xbar[i] * np.exp(-b*(pow(xbar[i],2) + pow(ybar[i],2)));
    return w

def save_paraview_results(w, n, m, t, dt, rank, size, cells, nodes):

    elements = {"triangle": cells}
    points = []
    for i in nodes:
        points.append([i[0], i[1], i[2]])

    data = {"h" : w.h, "u" : w.hu/w.h, "v": w.hv/w.h}  
    data = {"h": data, "u":data, "v": data}
    maxh = np.zeros(1)
    maxh = max(w.h)
    integral_sum = np.zeros(1)
    
    
    comm.Reduce(maxh, integral_sum, MPI.MAX, 0)
    if rank == 0:
        print(" **************************** Computing ****************************") 
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Iteration = ", n , "time = ", t, "time step = ", dt)
        print("max h =", integral_sum[0])
        
    meshio.write_points_cells("results/visu"+str(rank)+"-"+str(m)+".vtu", 
                              points, elements, cell_data=data, file_format="vtu")

    if(rank == 0 and size > 1 ):
        with open("results/visu"+str(m)+".pvtu", "a") as text_file:
            text_file.write("<?xml version=\"1.0\"?>\n")
            text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
            text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n") 
            text_file.write("<PPoints>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
            text_file.write("</PPoints>\n") 
            text_file.write("<PCells>\n") 
            text_file.write("<PDataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"offsets\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Int64\" Name=\"types\" format=\"binary\"/>\n")
            text_file.write("</PCells>\n") 
            text_file.write("<PCellData Scalars=\"h\">\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"h\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"v\" format=\"binary\"/>\n")
            text_file.write("</PCellData>\n")
            for i in range(size):
            	name1="visu";
            	bu1 = [10];
            	bu1 = str(i)
            	name1+=bu1;
            	name1+="-"+str(m);
            	name1+=".vtu";
            	text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")                    
            text_file.write("</PUnstructuredGrid>\n") 
            text_file.write("</VTKFile>")

        
        
