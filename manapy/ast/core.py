#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:53:35 2022

@author: kissami
"""

import numpy as np

from manapy.comms import all_to_all
from manapy.comms.pyccel_comm import define_halosend

from manapy.ast.pyccel_functions import (centertovertex_2d, centertovertex_3d, 
                                         ghost_value_neumann, ghost_value_dirichlet,
                                         ghost_value_slip, ghost_value_nonslip, 
                                         face_gradient_2d, face_gradient_3d,
                                         haloghost_value_dirichlet, haloghost_value_neumann, 
                                         haloghost_value_slip, haloghost_value_nonslip, 
                                         cell_gradient_2d, cell_gradient_3d, barthlimiter_2d, 
                                         barthlimiter_3d, facetocell)
 
from types import LambdaType
# from mpi4py import MPI
# from pyccel.ast.core import CodeBlock, FunctionDef
# from pyccel.codegen.printing.pycode import pycode



class Boundary():
    """ """
    def __init__(self, BCtype=None, BCvalueface = None, BCvaluenode = None, BCvaluehalo = None, BCloc = None, BCtypeindex = None, domain=None):
        
        if domain is None:
            raise ValueError("domain must be given")
            
        self._BCtype   = BCtype
        self.BCvalueface  = BCvalueface
        self.BCvaluenode  = BCvaluenode
        self.BCvaluehalo  = BCvaluehalo
        self._domain   = domain
        
        self._func_ghost_args = []
        self._func_haloghost_args = []
        
        if BCloc == "in":
            self._BCfaces  = self.domain._infaces
            self._BCnodes  = self.domain._innodes
            self._BCtypeindex  = BCtypeindex
        if BCloc == "out":
            self._BCfaces  = self.domain._outfaces
            self._BCnodes  = self.domain._outnodes
            self._BCtypeindex  = BCtypeindex
        if BCloc == "bottom":
            self._BCfaces  = self.domain._bottomfaces
            self._BCnodes  = self.domain._bottomnodes
            self._BCtypeindex  = BCtypeindex
        if BCloc == "upper":
            self._BCfaces  = self.domain._upperfaces
            self._BCnodes  = self.domain._uppernodes
            self._BCtypeindex  = BCtypeindex
        if BCloc == "front":
            self._BCfaces  = self.domain._frontfaces
            self._BCnodes  = self.domain._frontnodes
            self._BCtypeindex  = BCtypeindex
        if BCloc == "back":
            self._BCfaces  = self.domain._backfaces
            self._BCnodes  = self.domain._backnodes
            self._BCtypeindex  = BCtypeindex
        
        if self._BCtype == "neumann" or self._BCtype == "periodic":
            self._func_ghost = ghost_value_neumann
            self._func_haloghost = haloghost_value_neumann
        elif self._BCtype == "dirichlet":
            self._func_ghost = ghost_value_dirichlet
            self._func_haloghost = haloghost_value_dirichlet
        elif self._BCtype == "slip":
            self._func_ghost = ghost_value_slip
            self._func_haloghost = haloghost_value_slip
            self._func_ghost_args.extend([self._BCvaluefacetmp, self.domain.faces.normal, self.domain.faces.mesure])
            self._func_haloghost_args.extend([self._BCvaluehalotmp, self.domain.nodes.ghostfaceinfo])
        
        elif self._BCtype == "nonslip":
            self._func_ghost = ghost_value_nonslip
            self._func_haloghost = haloghost_value_nonslip
            
    @property
    def domain(self):
        return self._domain
    
    @property
    def BCfaces(self):
        return self._BCfaces
    
    @property
    def BCnodes(self):
        return self._BCnodes
    
    @property
    def BCtypeindex(self):
        return self._BCtypeindex

class Variable():
    
    """ """
    def __init__(self, domain=None, terms=None, comm=None, name=None, BC=None, values=None, *args, **kwargs):
        
       
        if domain is None:
            raise ValueError("domain must be given")
        
        self._domain = domain
        
        self._dim = self.domain.dim
        self._comm   = self.domain.halos.comm_ptr
        
        self._nbfaces = self.domain.nbfaces
        self._nbcells = self.domain.nbcells
        self._nbnodes = self.domain.nbnodes
        self._nbhalos = self.domain.nbhalos
        self._nbghost = self.domain.nbfaces
        
        self.cell = np.zeros(self.nbcells)
        self.node = np.zeros(self.nbnodes)
        self.face = np.zeros(self.nbfaces)
        self.ghost = np.zeros(self.nbfaces)
        self.halo = np.zeros(self.nbhalos)
        
        self.gradcellx = np.zeros(self._nbcells)
        self.gradcelly = np.zeros(self._nbcells)
        self.gradcellz = np.zeros(self._nbcells)
        
        self.gradhalocellx = np.zeros(self._nbhalos)
        self.gradhalocelly = np.zeros(self._nbhalos)
        self.gradhalocellz = np.zeros(self._nbhalos)
        
        self.gradfacex = np.zeros(self._nbfaces)
        self.gradfacey = np.zeros(self._nbfaces)
        self.gradfacez = np.zeros(self._nbfaces)
        
        
        self.psi = np.zeros(self._nbcells)
        self.psihalo = np.zeros(self._nbhalos)
        
        self.halotosend  = np.zeros(len(self.domain.halos.halosint))
        self.haloghost = np.zeros(self.domain.halos.sizehaloghost)
        
        self._name    = name
        
        
        
        if terms is not None:
            for i in terms:
                self.__dict__[i] = np.zeros(self.nbcells)
       
        if self.dim == 2:
            self._BCs = {"in":None, "out":None, "bottom":None, "upper":None}
            self._func_interp = centertovertex_2d
            self._cell_gradient = cell_gradient_2d
            self._face_gradient  = face_gradient_2d
            self._barthlimiter = barthlimiter_2d
            self._args_func = [self.domain.nodes.R_x, self.domain.nodes.R_y, self.domain.nodes.lambda_x, self.domain.nodes.lambda_y]
            
        elif self.dim == 3:
            self._BCs = {"in":None, "out":None, "bottom":None, "upper":None, "front":None, "back":None}
            self._typeOfCells = "tetra"
            self._func_interp = centertovertex_3d
            self._cell_gradient = cell_gradient_3d
            self._face_gradient  = face_gradient_3d
            self._barthlimiter = barthlimiter_3d
            self._args_func = [self.domain.nodes.R_x, self.domain.nodes.R_y, self.domain.nodes.R_z, self.domain.nodes.lambda_x, 
                              self.domain.nodes.lambda_y, self.domain.nodes.lambda_z]
            
            
        self.domain.Pbordnode = np.zeros(self.domain.nbnodes)
        self.domain.Pbordface = np.zeros(self.domain.nbfaces)
        
        self.dirichletfaces = []
        self.neumannfaces = []
        
        self.BCdirichlet = []
        self.BCneumann = []
        
        
        valueface = np.zeros(self.domain._nbfaces)
        valuenode = np.zeros(self.domain._nbnodes)
        valuehalo = np.zeros(self.domain.halos.sizehaloghost)
        
        if BC is None:
            for loc in self._BCs.keys():
                if self.domain._BCs[loc][0] == "periodic":
                    self.BCs[loc] = Boundary(BCtype = "periodic", BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCs[loc].BCvalueface = np.array([])
                    self.BCs[loc].BCvaluenode = np.array([])
                    self.BCs[loc].BCvaluehalo = np.array([])
                   
                else:
                    self.BCs[loc] = Boundary(BCtype = "neumann", BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                
                self.BCs[loc].BCvalueface = valueface
                self.BCs[loc].BCvaluenode = valuenode
                self.BCs[loc].BCvaluehalo = valuehalo
            
        else:
            for loc, bct in BC.items():
                if self.domain._BCs[loc][0] == "periodic":
                    if bct != "periodic":
                        raise ValueError("BC must be periodic for "+ str(loc))
                        
                if self.domain._BCs[loc][0] != "periodic":
                    if bct == "periodic":
                        raise ValueError("BC must be not periodic for "+ str(loc))
               
                if bct == "dirichlet":
                    
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCdirichlet.append(self.BCs[loc]._BCtypeindex)
                    self.dirichletfaces.extend(self.BCs[loc]._BCfaces)
                
                    if loc not in values.keys():
                        raise ValueError("Value of dirichlet BC for "+str(loc)+" faces must be given")
                    
                    #TODO check valuehalo (face center miss)
                    if isinstance(values[loc], LambdaType):
                        for i in self.BCs[loc]._BCfaces:
                            valueface[i] = values[loc](self.domain.faces.center[i][0], self.domain.faces.center[i][1], 
                                                        self.domain.faces.center[i][2])
                        for i in np.where(self.domain.nodes.oldname==self.BCs[loc]._BCtypeindex)[0]:
                            valuenode[i] = values[loc](self.domain.nodes.vertex[i][0], self.domain.nodes.vertex[i][1], 
                                                        self.domain.nodes.vertex[i][2])
                        
                            for j in range(len(self.domain.nodes.haloghostcenter[i])):
                                
                                cell = int(self.domain.nodes.haloghostcenter[i][j][-1])
                                if cell != -1:
                                    center = self.domain.nodes.haloghostfaceinfo[i][j][0:3]
                                    valuehalo[cell] = values[loc](center[0], center[1], center[2])
    
                    elif isinstance(values[loc], (int, float)):
                        for i in self.BCs[loc]._BCfaces:
                            valueface[i] = values[loc]
                            
                        for i in np.where(self.domain.nodes.oldname==self.BCs[loc]._BCtypeindex)[0]:
                            valuenode[i] = values[loc]
                        
                            for j in range(len(self.domain.nodes.haloghostcenter[i])):
                                    cell = int(self.domain.nodes.haloghostcenter[i][j][-1])
                                    if cell != -1:
                                        valuehalo[cell] = values[loc]
                                        
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
               
                elif bct == "neumann":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                    
                elif bct == "periodic":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCs[loc].BCvalueface = np.array([])
                    self.BCs[loc].BCvaluenode = np.array([])
                    self.BCs[loc].BCvaluehalo = np.array([])
                    
                    
                elif bct == "slip":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                    
                    
                elif bct == "nonslip":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                
        self._BCin     = self.BCs["in"]
        self._BCout    = self.BCs["out"]
        self._BCbottom = self.BCs["bottom"]
        self._BCupper  = self.BCs["upper"]
        
        # print(self.BCs)
        if self.dim == 3:
            self._BCfront = self.BCs["front"]
            self._BCback  = self.BCs["back"]
        elif self.dim == 2:
            self._BCfront = 0
            self._BCback  = 0
        
        self.dirichletfaces.sort()
        self.neumannfaces.sort()
        
        self.dirichletfaces = np.asarray(self.dirichletfaces, dtype=np.int64)
        self.neumannfaces = np.asarray(self.neumannfaces, dtype=np.int64)
        self.BCdirichlet = np.asarray(self.BCdirichlet)
        self.BCneumann = np.asarray(self.BCneumann)
        
    def update_values(self, value=None):
        
        self.update_halo_value()
        self.update_ghost_value()
        # self.interpolate_celltonode()
        
    def update_halo_value(self):
        #update the halo values
        define_halosend(self.cell, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.halo, self.comm)
        self.comm.Barrier()
    
    def interpolate_facetocell(self):
        facetocell(self.face, self.cell, self.domain.cells.faceid, self.dim)
    
    def interpolate_celltonode(self):
        
        self.update_halo_value()
        self.update_ghost_value()
        self._func_interp(self.cell, self.ghost, self.halo, self.haloghost, self.domain.cells.center, self.domain.halos.centvol, 
                          self.domain.nodes.cellid, self.domain.nodes.periodicid, self.domain.nodes.halonid, self.domain.nodes.vertex, 
                          self.domain.nodes.name, self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, *self._args_func, 
                          self.domain.nodes.number, self.domain.cells.shift, self.comm.Get_size(), self.node)
        
        
    def compute_cell_gradient(self):
        
        self.update_halo_value()
        self.update_ghost_value()
        
        self._cell_gradient(self.cell, self.ghost, self.halo, self.haloghost, self.domain.cells.center, self.domain.cells.cellnid, 
                            self.domain.cells.halonid, self.domain.cells.nodeid, self.domain.cells.periodicnid, self.domain.nodes.periodicid, 
                            self.domain.nodes.name, 
                            self.domain.nodes.ghostcenter,  self.domain.nodes.haloghostcenter, self.domain.nodes.vertex, self.domain.halos.centvol,
                            self.domain.cells.shift, self.comm.Get_size(), self.gradcellx, self.gradcelly, self.gradcellz)
        
        
       
        #The limiter depend on hc value
        self._barthlimiter(self.cell, self.ghost, self.halo, self.gradcellx, self.gradcelly, self.gradcellz,
                           self.psi, self.domain.faces.cellid, self.domain.cells.faceid, self.domain.faces.name, 
                           self.domain.faces.halofid, self.domain.cells.center, self.domain.faces.center)
        
        
        self.comm.Barrier()
        #update the halo values
        define_halosend(self.gradcellx, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.gradhalocellx, self.comm)
        
        #update the halo values
        define_halosend(self.gradcelly, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.gradhalocelly, self.comm)
        
        
        #update the halo values
        define_halosend(self.gradcellz, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.gradhalocellz, self.comm)
        
        #update the halo values
        define_halosend(self.psi, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.psihalo, self.comm)
        
    def compute_face_gradient(self):
        
        self.interpolate_celltonode()
        self._face_gradient(self.cell, self.ghost, self.halo, self.node, self.domain.faces.cellid, 
                            self.domain.faces.nodeid, self.domain.faces.ghostcenter, self.domain.faces.name, 
                            self.domain.faces.halofid, self.domain.cells.center,
                            self.domain.halos.centvol, self.domain.nodes.vertex, self.domain.faces.airDiamond,
                            self.domain.faces.normal, self.domain.faces.f_1, self.domain.faces.f_2, 
                            self.domain.faces.f_3, self.domain.faces.f_4, self.domain.cells.shift, 
                            self.gradfacex, self.gradfacey, self.gradfacez, self.domain._innerfaces,
                            self.domain.halofaces, self.dirichletfaces, self.neumannfaces, 
                            self.domain.periodicboundaryfaces)
        
    def update_ghost_value(self):
        for BC in self._BCs.values():
            BC._func_ghost(BC.BCvalueface, self.ghost, self.domain.faces.cellid, np.asarray(BC.BCfaces, dtype=np.int64))
            BC._func_haloghost(BC.BCvaluehalo, self.haloghost, self.domain.nodes.haloghostcenter, 
                               BC.BCtypeindex,  self.domain.halonodes) 
            
    def norml2(self, exact, order=None):
        
        if order is None:
            order = 1
        assert self.nbcells == len(exact), 'exact solution must have length of cells'
        
        Error = np.zeros(self.nbcells)
        Ex = np.zeros(self.nbcells)
       
        for i in range(len(exact)):
            Error[i] = np.fabs(self.cell[i] - exact[i]) * self.domain.cells.volume[i]
            Ex[i] = np.fabs(exact[i]) * self.domain.cells.volume[i]
    
        ErrorL2 = np.linalg.norm(Error,ord=order)/np.linalg.norm(Ex,ord=order)
        
        return ErrorL2
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def comm(self):
        return self._comm
    
    # @property
    # def cell(self):
    #     return self._cell
    
    # @property
    # def node(self):
    #     return self._node
    
    # @property
    # def face(self):
    #     return self._face
    
    # @property
    # def ghost(self):
    #     return self._ghost
    
    # @property
    # def psi(self):
    #     return self._psi
    
    
    # @property
    # def gradcellx(self):
    #     return self._gradcellx
    
    # @property
    # def gradcelly(self):
    #     return self._gradcellx
    
    # @property
    # def gradcellz(self):
    #     return self._gradcellz
    
    # @property
    # def psihalo(self):
    #     return self._psihalo
    
    # @property
    # def gradhalocellx(self):
    #     return self._gradhalocellx
    
    # @property
    # def gradhalocelly(self):
    #     return self._gradhalocelly
    
    # @property
    # def gradhalocellz(self):
    #     return self._gradhalocellz
    # @property
    # def gradface(self):
    #     return self._gradface
    
    # @property
    # def halotosend(self):
    #     return self._halotosend
    
    # @property
    # def halotorecv(self):
    #     return self._halotorecv
    
    # @property
    # def halo(self):
    #     return self._halo
    
    # @property
    # def haloghost(self):
    #     return self._haloghost
    
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def nbhalos(self):
        return self._nbhalos
    
    @property
    def name(self):
        return self._name
    
    @property
    def BCs(self):
        return self._BCs
    
    @property
    def BCin(self):
        return self._BCin
    
    @property
    def BCout(self):
        return self._BCout
    
    @property
    def BCupper(self):
        return self._BCupper
    
    @property
    def BCbottom(self):
        return self._BCbottom
    
    @property
    def BCback(self):
        return self._BCback
    
    @property
    def BCfront(self):
        return self._BCfront
    
    # @property
    # def func_interp(self):
    #     return self._func_interp
    
    # @property
    # def cell_gradient(self):
    #     return self._cell_gradient
    
    # @property
    # def face_gradient(self):
    #     return self._face_gradient
    
    # @property
    # def barthlimiter(self):
    #     return self._barthlimiter
    
    # @property
    # def args_func(self):
    #     return self._args_func
    
    # @property
    # def face_gradient(self):
    #     return self._face_gradient
    
