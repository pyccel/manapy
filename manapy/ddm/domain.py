#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:05:57 2022

@author: kissami
"""

from numpy import asarray, int64, double, zeros, where, unique, sort, ones, array, concatenate
import meshio

from manapy.ddm.pyccel_ddm import variables, face_gradient_info_2d, variables_3d
from manapy.ddm.numba_ddm import face_gradient_info_3d, read_mesh_file

from manapy.comms import update_haloghost_info_2d, update_haloghost_info_3d, prepare_comm
from mpi4py import MPI

from manapy.ddm.pyccel_ddm import (create_info_2dfaces, create_info_3dfaces,
                                   Compute_2dcentervolumeOfCell, Compute_3dcentervolumeOfCell,
                                   create_cellsOfFace, create_2dfaces, create_cell_faceid,
                                   create_3dfaces, create_NormalFacesOfCell)

from manapy.ddm.numba_ddm import (create_2doppNodeOfFaces, create_3doppNodeOfFaces, update_pediodic_info_2d,
                                  create_NeighborCellByFace, create_node_cellid, update_pediodic_info_3d,
                                  oriente_3dfacenodeid, create_2d_halo_structure, create_3d_halo_structure)

class Cell():
    """ """
    __slots__=['_nbcells', '_nodeid', '_faceid', '_cellfid', '_cellnid', '_halonid', '_center',
               '_volume', '_nf', '_globtoloc', '_loctoglob', '_tc', '_periodicnid', '_periodicfid', '_shift']
    def __init__(self):
        pass
        
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nodeid(self):
        return self._nodeid
    
    @property
    def faceid(self):
        return self._faceid
    
    @property
    def cellfid(self):
        return self._cellfid
    
    @property
    def cellnid(self):
        return self._cellnid
    
    @property
    def halonid(self):
        return self._halonid
    
    @property
    def center(self):
        return self._center
    
    @property
    def volume(self):
        return self._volume
    
    @property
    def nf(self):
        return self._nf
    
    @property
    def globtoloc(self):
        return self._globtoloc
    
    @property
    def loctoglob(self):
        return self._loctoglob
    
    @property
    def tc(self):
        return self._tc
    
    @property
    def periodicnid(self):
        return self._periodicnid
    
    @property
    def periodicfid(self):
        return self._periodicfid
    
    @property
    def shift(self):
        return self._shift
        
        
class Node():
    """ """
    __slots__= ['_nbnodes', '_vertex', '_name', '_oldname', '_cellid', '_ghostcenter', '_haloghostcenter', '_ghostfaceinfo', 
                '_haloghostfaceinfo', '_loctoglob', '_halonid', '_nparts', '_periodicid', '_R_x', '_R_y', '_R_z', '_number', 
                '_lambda_x', '_lambda_y', '_lambda_z']
     
    def __init__(self, nbnodes=None):
        pass
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def vertex(self):
        return self._vertex
    
    @property
    def name(self):
        return self._name
    
    @property
    def oldname(self):
        return self._oldname
    
    @property
    def cellid(self):
        return self._cellid
    
    @property
    def ghostcenter(self):
        return self._ghostcenter
    
    @property
    def haloghostcenter(self):
        return self._haloghostcenter
    
    @property
    def ghostfaceinfo(self):
        return self._ghostfaceinfo
    
    @property
    def haloghostfaceinfo(self):
        return self._haloghostfaceinfo
    
    @property
    def loctoglob(self):
        return self._loctoglob
    
    
    @property
    def halonid(self):
        return self._halonid
    
    @property
    def nparts(self):
        return self._nparts
    
    @property
    def periodicid(self):
        return self._periodicid
    
    @property
    def R_x(self):
        return self._R_x
    
    @property
    def R_y(self):
        return self._R_y
    
    @property
    def R_z(self):
        return self._R_z
    
    @property
    def number(self):
        return self._number
    
    @property
    def lambda_x(self):
        return self._lambda_x
    
    @property
    def lambda_y(self):
        return self._lambda_y
    
    @property
    def lambda_z(self):
        return self._lambda_z

class Face():
    """ """
    __slots__= ['_nbfaces', '_nodeid', '_cellid', '_name', '_normal', '_mesure', '_center', '_ghostcenter', '_oppnodeid', '_halofid',
                '_halofid', '_param1', '_param2', '_param3', '_param4', '_f_1', '_f_2', '_f_3', '_f_4', '_airDiamond']
                
    def __init__(self):
        pass
        
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nodeid(self):
        return self._nodeid
    
    @property
    def cellid(self):
        return self._cellid
    
    @property
    def name(self):
        return self._name
    
    @property
    def normal(self):
        return self._normal
    
    @property
    def mesure(self):
        return self._mesure
    
    @property
    def center(self):
        return self._center
    
    
    @property
    def ghostcenter(self):
        return self._ghostcenter
    
    
    @property
    def oppnodeid(self):
        return self._oppnodeid
    
    @property
    def halofid(self):
        return self._halofid
    
    @property
    def param1(self):
        return self._param1
    
    @property
    def param2(self):
        return self._param2
    
    @property
    def param3(self):
        return self._param3
    
    @property
    def param4(self):
        return self._param4
    
    @property
    def f_1(self):
        return self._f_1
    
    @property
    def f_2(self):
        return self._f_2
    
    @property
    def f_3(self):
        return self._f_3
    
    @property
    def f_4(self):
        return self._f_4
    
    @property
    def airDiamond(self):
        return self._airDiamond
    
    # @property
    # def K(self):
    #     return self._K
        
class Halo():
    """ """
    __slots__= ['_halosint', '_halosext', '_neigh', '_centvol', '_faces', '_nodes', '_sizehaloghost', '_scount', '_rcount', '_sdepl', 
                '_rdepl','_indsend', '_comm_ptr']
    
    def __init__(self):
        pass
        
    @property
    def halosint(self):
        return self._halosint
    
    @property
    def halosext(self):
        return self._halosext
    
    @property
    def neigh(self):
        return self._neigh
    
    @property
    def centvol(self):
        return self._centvol
    
    @property
    def faces(self):
        return self._faces
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def sizehaloghost(self):
        return self._sizehaloghost
    
    @property
    def scount(self):
        return self._scount
    
    @property
    def rcount(self):
        return self._rcount
    
    @property
    def indsend(self):
        return self._indsend
    
    
    @property
    def comm_ptr(self):
        return self._comm_ptr
        
class Domain():
    """ """

    def __init__(self, dim=None, comm=None):
        
        if comm is None:
            comm = MPI.COMM_WORLD
        
        if dim is None:
            raise ValueError("dim file must be given")
        
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        self._dim = dim
        self._comm = comm
        #read nodes.vertex, cells.tc, cells.nodeid
        tc, nodeid, vertex, halosint, halosext, centvol, Cellglobtoloc, Cellloctoglob, Nodeloctoglob, neigh, nparts = read_mesh_file(size, rank)
        # tc, nodeid, vertex =  read_mesh_file(size, rank)
        
        self._nbcells = len(nodeid)
        self._nbnodes = len(vertex)
        
        nodes = Node()
        cells = Cell()
        faces = Face()
        halos = Halo()
        
        nodes._vertex = asarray(vertex, dtype=double)
        nodes._loctoglob = asarray(Nodeloctoglob, dtype=int64)
        nodes._nparts = nparts
        
        
        cells._nodeid = asarray(nodeid, dtype=int64)
        cells._tc = asarray(tc, dtype=int64)
        cells._loctoglob = asarray(Cellloctoglob, dtype=int64)
        cells._globtoloc = Cellglobtoloc
        
        halos._neigh = asarray(neigh, dtype=int64)
        halos._halosint = asarray(halosint, dtype=int64)
        halos._halosext = asarray(halosext, dtype=int64)
        halos._centvol = asarray(centvol, dtype=double)
        
        nodes._name = zeros(self.nbnodes, dtype=int64)
        for i in range(self.nbnodes):
            nodes._name[i] = int(nodes._vertex[i][3])
        
        #Create center and volume for each cell (pycceliser)
        cells._center = zeros((self.nbcells, 3), dtype=double)
        cells._volume = zeros(self.nbcells, dtype=double)
        
        if dim == 2:
            Compute_2dcentervolumeOfCell(cells._nodeid, nodes._vertex, self.nbcells, cells._center, cells._volume)
        elif dim == 3:
            Compute_3dcentervolumeOfCell(cells._nodeid, nodes._vertex, self.nbcells, cells._center, cells._volume)
        #create cells over each node (still numba function)
        nodes._cellid, cells._cellnid =  create_node_cellid(cells._nodeid, nodes._vertex, self.nbcells, self.nbnodes, dim=self.dim)
        
        #creating faces (pycceliser)
        p_faces = zeros(((self.dim+1)*self.nbcells, self.dim), dtype=int64)
        cellf = zeros((self.nbcells, self.dim+1), dtype=int64)
        
        if self.dim == 2:
            self._typeOfCells = "triangle"
            create_2dfaces(cells._nodeid, self.nbcells, p_faces, cellf)
        elif self.dim == 3:
            self._typeOfCells = "tetra"
            create_3dfaces(cells._nodeid, self.nbcells, p_faces, cellf)
        
        faces._nodeid, oldTonewIndex = unique(sort(p_faces), axis=0, return_inverse=True)
        
        self._nbfaces = len(faces._nodeid)
        
        cells._faceid = zeros((self.nbcells, (self.dim+1)), dtype=int64)
        create_cell_faceid(self.nbcells, oldTonewIndex, cellf, cells._faceid, dim=self.dim)
        
        ############################################################################
        #creater cells left and right of each face (pycceliser)
        faces._cellid = -1*ones((self.nbfaces, 2), dtype=int64)
        create_cellsOfFace(cells._faceid, self.nbcells, self.nbfaces, faces._cellid, dim=self.dim)
        ############################################################################
        cells._cellfid = create_NeighborCellByFace(cells._faceid, faces._cellid, self.nbcells, dim=self.dim)
        
        ############################################################################
        #create info of faces (pycceliser)
        faces._name   = zeros(self.nbfaces, dtype=int64)
        faces._normal = zeros((self.nbfaces, 3), dtype=double)
        faces._mesure = zeros(self.nbfaces, dtype=double)
        faces._center = zeros((self.nbfaces, 3), dtype=double)
        
        if dim == 2:
            create_info_2dfaces(faces._cellid, faces._nodeid, nodes._name, nodes._vertex, cells._center, 
                                self.nbfaces, faces._normal, faces._mesure, faces._center, faces._name)
        elif dim == 3:
            create_info_3dfaces(faces._cellid, faces._nodeid, nodes._name, nodes._vertex, cells._center, 
                                self.nbfaces, faces._normal, faces._mesure, faces._center, faces._name)
    
        ############################################################################
        #Create outgoing normal vectors (pycceliser)
        cells._nf = zeros((self.nbcells, self.dim+1, 3), dtype=double)
        if self.dim == 2:
            create_NormalFacesOfCell(cells._center, faces._center, cells._faceid, faces._normal, \
                                     self.nbcells, cells._nf, dim=self.dim)
        ###########################################################################   
        #still numba function
        if self.dim == 2:
            faces._oppnodeid = create_2doppNodeOfFaces(cells._nodeid, cells._faceid, faces._nodeid, self.nbcells, self.nbfaces)
        elif dim == 3:
            #TODO active if needed
            faces._oppnodeid = create_3doppNodeOfFaces(cells._nodeid, cells._faceid, faces._nodeid, self.nbcells, self.nbfaces)
            #pycceliser (not yet)
            faces._nodeid = oriente_3dfacenodeid(faces._nodeid, faces._normal, nodes._vertex)
    
        if self.dim == 2:
            create_2d_halo_structure(cells, faces, nodes, halos, size, self.nbcells, self.nbfaces, self.nbnodes)
        elif self.dim == 3:
            create_3d_halo_structure(cells, faces, nodes, halos, size, self.nbcells, self.nbfaces, self.nbnodes)
    
        # #compute K
        # faces._K = zeros((self.nbfaces, 3))
        # compute_K(faces._cellid, faces._name, faces._ghostcenter, cells._center, faces._K, dim=self.dim)
        
        #setting old name
        nodes._oldname = zeros(self.nbnodes, dtype=int64)
        nodes._oldname[:] = nodes._vertex[:,3]
        
        #######################################################################################################
        #compute the arrays needed for the mpi communication
        halos._scount, halos._rcount, halos._indsend, self._nbhalos, halos._comm_ptr = prepare_comm(cells, halos)
        
        ###################################Update faces and names type##########################################
        self._innerfaces  = where(faces._name==0)[0]
        self._infaces     = where(faces._name==1)[0]
        self._outfaces    = where(faces._name==2)[0]
        self._upperfaces  = where(faces._name==3)[0]
        self._bottomfaces = where(faces._name==4)[0]
        
        self._periodicinfaces     = where(faces._name==11)[0]
        self._periodicoutfaces    = where(faces._name==22)[0]
        self._periodicupperfaces  = where(faces._name==33)[0]
        self._periodicbottomfaces = where(faces._name==44)[0]
            
        self._halofaces   = where(faces._name==10)[0]
      
        self._boundaryfaces = concatenate([self._infaces, self._outfaces, self._bottomfaces, self._upperfaces] )
        self._periodicboundaryfaces = concatenate([self._periodicinfaces, self._periodicoutfaces, self._periodicbottomfaces, self._periodicupperfaces] )
        
        
        self._innernodes  = where(nodes._name==0)[0]
        self._innodes     = where(nodes._name==1)[0]
        self._outnodes    = where(nodes._name==2)[0]
        self._uppernodes  = where(nodes._name==3)[0]
        self._bottomnodes = where(nodes._name==4)[0]
        
        self._periodicinnodes     = where(nodes._name==11)[0]
        self._periodicoutnodes    = where(nodes._name==22)[0]
        self._periodicuppernodes  = where(nodes._name==33)[0]
        self._periodicbottomnodes = where(nodes._name==44)[0]
        
        self._halonodes   = where(nodes._name==10)[0]
        
        self._boundarynodes = concatenate([self._innodes, self._outnodes, self._bottomnodes, self._uppernodes] )
        self._periodicboundarynodes = concatenate([self._periodicinnodes, self._periodicoutnodes, self._periodicbottomnodes, self._periodicuppernodes] )
        
        if self.dim == 3:
            
            self._frontfaces    = where(faces._name==5)[0]
            self._backfaces     = where(faces._name==6)[0]
            self._periodicfrontfaces  = where(faces._name==55)[0]
            self._periodicbackfaces   = where(faces._name==66)[0]

            self._frontnodes    = where(nodes._name==5)[0]
            self._backnodes     = where(nodes._name==6)[0]
            self._periodicfrontnodes    = where(nodes._name==55)[0]
            self._periodicbacknodes     = where(nodes._name==66)[0]

            self._boundaryfaces = concatenate([self._boundaryfaces, self._backfaces, self._frontfaces] )
            self._periodicboundaryfaces = concatenate([self._periodicboundaryfaces, self._periodicbackfaces, self._periodicfrontfaces] )
            
            
            self._boundarynodes = concatenate([self._boundarynodes, self._backnodes, self._frontnodes] )
            self._periodicboundarynodes = concatenate([self._periodicboundarynodes, self._periodicbacknodes, self._periodicfrontnodes] )
            
        self._boundaryfaces = sort(self._boundaryfaces)
        self._periodicboundaryfaces = sort(self._periodicboundaryfaces)
        self._boundarynodes = sort(self._boundarynodes)
        self._periodicboundarynodes = sort(self._periodicboundarynodes)           
        #########################################################################################################
        
        #update periodic boundaries
        if self.dim == 2:
            update_pediodic_info_2d(nodes, cells, faces, halos, self.nbnodes, self.nbcells, 
                                    self._periodicinfaces, self._periodicoutfaces, self._periodicupperfaces, self._periodicbottomfaces,
                                    self._periodicinnodes, self._periodicoutnodes, self._periodicuppernodes, self._periodicbottomnodes)
        elif self.dim == 3:
            update_pediodic_info_3d(nodes, cells, faces, halos, self.nbnodes, self.nbcells, 
                                    self._periodicinfaces, self._periodicoutfaces, self._periodicupperfaces, self._periodicbottomfaces,
                                    self._periodicfrontfaces, self._periodicbackfaces, 
                                    self._periodicinnodes, self._periodicoutnodes, self._periodicuppernodes, self._periodicbottomnodes,
                                    self._periodicfrontnodes, self._periodicbacknodes)
        
        nodes._R_x = zeros(self.nbnodes, dtype=double)
        nodes._R_y = zeros(self.nbnodes, dtype=double)
        nodes._lambda_x = zeros(self.nbnodes, dtype=double)
        nodes._lambda_y = zeros(self.nbnodes, dtype=double)
        nodes._number = zeros(self.nbnodes, dtype=int64)
        
        faces._airDiamond = zeros(self.nbfaces, dtype=double)
        faces._param1 = zeros(self.nbfaces, dtype=double)
        faces._param2 = zeros(self.nbfaces, dtype=double)
        faces._param3 = zeros(self.nbfaces, dtype=double)
        
        faces._f_1  = zeros((self.nbfaces, self.dim), dtype=double)
        faces._f_2  = zeros((self.nbfaces, self.dim), dtype=double)
        faces._f_3  = zeros((self.nbfaces,dim), dtype=double)
        faces._f_4  = zeros((self.nbfaces,dim), dtype=double)
        
        self._BCs = {"in":["neumann", 1], "out":["neumann", 2],  "upper":["neumann", 3], "bottom":["neumann", 4]}
       
        if len(self._periodicinfaces) != 0:
             self._BCs["in"]  =  ["periodic", 11]
             self._BCs["out"] =  ["periodic", 22]
                
        if len(self._periodicupperfaces) != 0:
            self._BCs["bottom"] = ["periodic", 44]
            self._BCs["upper"] = ["periodic", 33]
        
        if self._dim == 2:

            faces._param4 = zeros(self.nbfaces, dtype=double)
           
            halos._sizehaloghost = update_haloghost_info_2d(nodes, cells, halos, self.nbnodes, self.halonodes)
        
            variables(cells._center, nodes._cellid, nodes._halonid, nodes._periodicid, nodes._vertex, nodes._name, 
                      nodes._ghostcenter, nodes._haloghostcenter, halos._centvol,  size, nodes._R_x, nodes._R_y,
                      nodes._lambda_x, nodes._lambda_y, nodes._number, cells._shift)
            
            face_gradient_info_2d(faces._cellid, faces._nodeid, faces._ghostcenter, faces._name, faces._normal, 
                                  cells._center, halos._centvol, faces._halofid, nodes._vertex, faces._airDiamond, 
                                  faces._param1, faces._param2, faces._param3, faces._param4, faces._f_1, 
                                  faces._f_2, faces._f_3, faces._f_4, cells._shift, self._dim)
        
        elif self._dim == 3:
            
            self._BCs["front"] = ["neumann", 5]
            self._BCs["back"]  = ["neumann", 6]
            
            if len(self._periodicfrontfaces) != 0:
                 self._BCs["front"] = ["periodic", 55]
                 self._BCs["back"] = ["periodic", 66]
            
            nodes._R_z = zeros(self.nbnodes, dtype=double)
            nodes._lambda_z = zeros(self.nbnodes, dtype=double)
            
            halos._sizehaloghost = update_haloghost_info_3d(nodes, cells, halos, self.nbnodes, self.halonodes)
            
            variables_3d(cells._center, nodes._cellid, nodes._halonid, nodes._periodicid, nodes._vertex, nodes._name, 
                         nodes._ghostcenter, nodes._haloghostcenter, halos._centvol, size,
                         nodes._R_x, nodes._R_y, nodes._R_z, nodes._lambda_x, nodes._lambda_y, 
                         nodes._lambda_z, nodes._number, cells._shift)

            face_gradient_info_3d(faces._cellid, faces._nodeid, faces._ghostcenter, faces._name,
                                  faces._normal, faces._mesure, cells._center, halos._centvol, 
                                  faces._halofid, nodes._vertex, faces._airDiamond, faces._param1, 
                                  faces._param2, faces._param3, faces._f_1, faces._f_2, cells._shift, self._dim)
        
        self._faces = faces
        self._cells = cells
        self._nodes = nodes
        self._halos = halos
        
    def __str__(self):
        """Pretty printing"""
        txt = '\n'
        txt += '> dim   :: {dim}\n'.format(dim=self.dim)
        txt += '> total cells  :: {cells}\n'.format(cells=self.nbcells)
        txt += '> total nodes  :: {nodes}\n'.format(nodes=self.nbnodes)
        txt += '> total faces  :: {faces}\n'.format(faces=self.nbfaces)

        return txt
    
    def save_on_cell(self, dt=0, time=0, niter=0, miter=0, value=None):
        
        if value is None:
            raise ValueError("value must be given")
        assert len(value) == self.nbcells, 'value size != number of cells'
       
        elements = {self.typeOfCells: self.cells._nodeid}
        points = self._nodes._vertex[:, :3]
        points = array(points)
        
        
        data = {"w" : value}
        data = {"w": data}
        
        maxw = max(value)
        
        integral_maxw = zeros(1)
    
        self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)
      
        if self.comm.rank == 0:
            print(" **************************** Computing ****************************")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Iteration = ", niter, "time = ", time, "time step = ", dt)
            print("max w =", integral_maxw[0])
    
        meshio.write_points_cells("results/visu"+str(self.comm.rank)+"-"+str(miter)+".vtu",
                                  points, elements, cell_data=data, file_format="vtu")
    
        if(self.comm.rank == 0 and self.comm.size > 1):
            with open("results/visu"+str(miter)+".pvtu", "a") as text_file:
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
                text_file.write("<PDataArray type=\"Float64\" Name=\"w\" format=\"binary\"/>\n")
                text_file.write("</PCellData>\n")

                for i in range(self.comm.size):
                    name1 = "visu"
                    bu1 = [10]
                    bu1 = str(i)
                    name1 += bu1
                    name1 += "-"+str(miter)
                    name1 += ".vtu"
                    text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
                text_file.write("</PUnstructuredGrid>\n")
                text_file.write("</VTKFile>")
                
    def save_on_node(self, dt=0, time=0, niter=0, miter=0, value=None):
        
        if value is None:
            raise ValueError("value must be given")
        assert len(value) == self.nbnodes, 'value size != number of nodes'
        
        elements = {self.typeOfCells: self.cells._nodeid}
        points = self._nodes._vertex[:, :3]
        
        data = {"w" : value}
        
        maxw = max(value)
        
        integral_maxw = zeros(1)
    
        self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)
      
        if self.comm.rank == 0:
            print(" **************************** Computing ****************************")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Iteration = ", niter, "time = ", time, "time step = ", dt)
            print("max w =", integral_maxw[0])
    
        meshio.write_points_cells("results/visu"+str(self.comm.rank)+"-"+str(miter)+".vtu",
                                  points, elements, point_data=data, file_format="vtu")
    
        if(self.comm.rank == 0 and self.comm.size > 1):
            with open("results/visu"+str(miter)+".pvtu", "a") as text_file:
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
                text_file.write("<PPointData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"w\" format=\"binary\"/>\n")
                text_file.write("</PPointData>\n")

                for i in range(self.comm.size):
                    name1 = "visu"
                    bu1 = [10]
                    bu1 = str(i)
                    name1 += bu1
                    name1 += "-"+str(miter)
                    name1 += ".vtu"
                    text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
                text_file.write("</PUnstructuredGrid>\n")
                text_file.write("</VTKFile>")
    
    @property
    def cells(self):
        return self._cells

    @property
    def faces(self):
        return self._faces

    @property
    def nodes(self):
        return self._nodes

    @property
    def halos(self):
        return self._halos

    @property
    def dim(self):
        return self._dim
    
    @property
    def comm(self):
        return self._comm
    
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nbhalos(self):
        return self._nbhalos
    
    @property
    def innerfaces(self):
        return self._innerfaces
    
    @property
    def infaces(self):
        return self._infaces
    
    @property
    def outfaces(self):
        return self._outfaces
    
    @property
    def bottomfaces(self):
        return self._bottomfaces
    
    @property
    def upperfaces(self):
        return self._upperfaces
    
    @property
    def halofaces(self):
        return self._halofaces
    
    @property
    def innernodes(self):
        return self._innernodes
    
    @property
    def innodes(self):
        return self._innodes
    
    @property
    def outnodes(self):
        return self._outnodes
    
    @property
    def bottomnodes(self):
        return self._bottomnodes
    
    @property
    def uppernodes(self):
        return self._uppernodes
    
    @property
    def halonodes(self):
        return self._halonodes
    
    @property
    def boundaryfaces(self):
        return self._boundaryfaces
    
    @property
    def boundarynodes(self):
        return self._boundarynodes
    
    @property
    def periodicboundaryfaces(self):
        return self._periodicboundaryfaces
    
    @property
    def periodicboundarynodes(self):
        return self._periodicboundarynodes
    
    @property
    def typeOfCells(self):
        return self._typeOfCells
    
