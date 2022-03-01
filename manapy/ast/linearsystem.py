#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:54:55 2022

@author: kissami
"""

from manapy.ast.pyccel_functions import (convert_solution, rhs_value_dirichlet_node, 
                                         get_triplet_3d, get_triplet_2d, compute_2dmatrix_size,
                                         compute_3dmatrix_size, 
                                         get_rhs_loc_2d, get_rhs_glob_2d,
                                         get_rhs_loc_3d, get_rhs_glob_3d, 
                                         compute_P_gradient_2d, compute_P_gradient_3d,
                                         rhs_value_dirichlet_face)

# from manapy.ast.pyccel_functions import get_triplet_3d

from manapy.ast import Variable
import numpy as np
from mpi4py import MPI

import mumps

class LinearSystem():
    """ """
    def __init__(self,  domain=None, var=None, solver=None, rhs=None, comm=None, debug=None):
        
        if comm is None:
            comm = MPI.COMM_WORLD
        if domain is None:
            raise ValueError("domain must be given")
        if var is None:
            raise ValueError("variable must be given")
        assert(isinstance(var, Variable)), "Variable must be given"
        
        
        self._domain = domain
        self._comm = comm
        self._var = var
        self._dim = self.domain.dim
        self.rhs = rhs
        self._solver = solver
        
        self.domain.Pbordnode = np.zeros(self.domain.nbnodes)
        self.domain.Pbordface = np.zeros(self.domain.nbfaces)
        
        matrixinnerfaces = np.concatenate([self.domain._innerfaces, self.domain._periodicinfaces, self.domain._periodicupperfaces])
        if self.dim == 3:
            matrixinnerfaces = np.concatenate([matrixinnerfaces, self.domain._periodicfrontfaces])
        self._matrixinnerfaces = np.sort(matrixinnerfaces)

        for BC in self.var.BCs.values():
            if BC._BCtype == "dirichlet":
                rhs_value_dirichlet_face(self.domain.Pbordface, np.asarray(BC.BCfaces, dtype=np.int64), BC.BCvalueface)
                rhs_value_dirichlet_node(self.domain.Pbordnode, np.where(self.domain.nodes.oldname==BC.BCtypeindex)[0],
                                         BC.BCvaluenode)
            
            if BC._BCtype == "neumann":
                for i in np.where(self.domain.nodes.oldname==BC.BCtypeindex)[0]:
                    self.domain.Pbordnode[i] = 1.
                    
        if self.dim == 2:
            self._compute_P_gradient = compute_P_gradient_2d
            self.dataSize =  compute_2dmatrix_size(self.domain.faces.nodeid, self.domain.faces.halofid, self.domain.nodes.cellid, 
                                                   self.domain.nodes.halonid, self.domain.nodes.periodicid,
                                                   self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, self.domain.nodes.oldname,
                                                   self.var.BCdirichlet, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
        elif self.dim == 3:
            self._compute_P_gradient = compute_P_gradient_3d
            self.dataSize =  compute_3dmatrix_size(self.domain.faces.nodeid, self.domain.faces.halofid, self.domain.nodes.cellid, 
                                                   self.domain.nodes.halonid, self.domain.nodes.periodicid,
                                                   self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, self.domain.nodes.oldname,
                                                   self.var.BCdirichlet, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
        

        self._row = np.zeros(self.dataSize, dtype=np.int32)
        self._col = np.zeros(self.dataSize, dtype=np.int32)
        self._data = np.zeros(self.dataSize, dtype=np.double)
        
        self._localsize  = self.domain.nbcells
        self._globalsize = comm.allreduce(self.localsize, op=MPI.SUM)
        
        self.x1converted = np.zeros(self.globalsize)

        if self.solver is None:
            self.solver = "spsolve"
            
        if self.solver == "spsolve":
            assert self.comm.Get_size() == 1, "spsolve is sequential choose either mumps or petsc as solver"
           
            self.rhs0_glob = np.zeros(self.globalsize)
            self.compute_global_rhs()
            
            if self.rhs is not None:
                assert len(self.rhs) == len(self.rhs0_glob), 'rhs must have global shape for mumps'
                self.rhs0_glob += self.rhs

        elif self.solver == "umfpack":
            assert self.comm.Get_size() == 1, "umfpack is sequential choose either mumps or petsc as solver"
            self.rhs0_glob = np.zeros(self.globalsize)
            self.compute_global_rhs()
        
        elif self.solver == "petsc":
            import petsc4py
            import sys
            petsc4py.init(sys.argv)
            from petsc4py import PETSc
            
            self.opts = PETSc.Options()

            self.ksp = PETSc.KSP()
            self.ksp.create()
        
            self.rhs0_loc = np.zeros(self.localsize)
            self.compute_local_rhs()
            
            if self.rhs is not None:
                assert len(self.rhs) == len(self.rhs0_loc), 'rhs must have global shape for mumps'
                self.rhs0_loc += self.rhs
                
        elif self.solver == "mumps":
            #Using local matrix and global rhs
            self.ctx = mumps.DMumpsContext(comm=self.comm)
            self.ctx.set_shape(self.localsize)
            if debug is None:
                #silent
                self.ctx.set_silent()
            
            self.rhs0_glob = np.zeros(self.globalsize)
            self.compute_global_rhs()
            self.rhs0_glob = self.comm.reduce(self.rhs0_glob, op=MPI.SUM, root=0)
            
            if self.rhs is not None:
                assert len(self.rhs) == len(self.rhs0_glob), 'rhs must have global shape for mumps'
                self.rhs0_glob += self.rhs
            
        self.sendcounts1 = np.array(self.comm.gather(self.localsize, root=0))
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def var(self):
        return self._var
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def row(self):
        return self._row
    
    @property
    def col(self):
        return self._col
    
    @property
    def data(self):
        return self._data
    
    @property
    def localsize(self):
        return self._localsize
    
    @property
    def globalsize(self):
        return self._globalsize
    
    @property
    def matrixinnerfaces(self):
        return self._matrixinnerfaces
    
    @property
    def solver(self):
        return self._solver
    
    @property
    def comm(self):
        return self._comm
    
    def compute_Sol_gradient(self):
     
        self.var.interpolate_celltonode()
        self._compute_P_gradient(self.var.cell, self.var.ghost, self.var.halo, self.var.node, self.domain.faces.cellid, 
                                 self.domain.faces.nodeid, self.domain.faces.ghostcenter, self.domain.faces.name, 
                                 self.domain.faces.halofid, self.domain.cells.center,
                                 self.domain.halos.centvol, self.domain.nodes.oldname, self.domain.faces.airDiamond,
                                 self.domain.faces.f_1, self.domain.faces.f_2,  self.domain.faces.f_3, self.domain.faces.f_4, 
                                 self.domain.faces.normal, self.domain.cells.shift, self.domain.Pbordnode, self.domain.Pbordface, 
                                 self.var.gradfacex, self.var.gradfacey, self.var.gradfacez, self.var.BCdirichlet, 
                                 self.domain.innerfaces, self.domain.halofaces, self.var.neumannfaces, 
                                 self.var.dirichletfaces, self.domain.periodicboundaryfaces)

    
    def compute_local_rhs(self):
        if self.dim == 2:
            get_rhs_loc_2d(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname, 
                           self.domain.cells.volume, self.domain.nodes.ghostcenter, self.domain.faces.param1, self.domain.faces.param2, 
                           self.domain.faces.param3, self.domain.faces.param4, self.domain.Pbordnode, self.domain.Pbordface, self.rhs0_loc, 
                           self.var.BCdirichlet, self.domain.faces.ghostcenter,  
                           self._matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
            
        elif self.dim == 3:
             get_rhs_loc_3d(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname,
                            self.domain.cells.volume, self.domain.nodes.ghostcenter, self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3,
                            self.domain.Pbordnode, self.domain.Pbordface, self.rhs0_loc, self.var.BCdirichlet, 
                            self._matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
             
    def compute_global_rhs(self):
        if self.dim == 2:
            get_rhs_glob_2d(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname,
                            self.domain.cells.volume, self.domain.nodes.ghostcenter, self.domain.cells.loctoglob, self.domain.faces.param1, 
                            self.domain.faces.param2, self.domain.faces.param3, self.domain.faces.param4, self.domain.Pbordnode, self.domain.Pbordface, 
                            self.rhs0_glob, self.var.BCdirichlet, self.domain.faces.ghostcenter, 
                            self._matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
            
        elif self.dim == 3:
            get_rhs_glob_3d(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname,
                            self.domain.cells.volume, self.domain.nodes.ghostcenter, self.domain.cells.loctoglob, self.domain.faces.param1, 
                            self.domain.faces.param2, self.domain.faces.param3, self.domain.Pbordnode, self.domain.Pbordface, self.rhs0_glob, 
                            self.var.BCdirichlet, self._matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
            
    def assembly(self):
        
        if self.dim == 2:
            get_triplet_2d(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.vertex, 
                           self.domain.faces.halofid, self.domain.halos.halosext, self.domain.nodes.name, self.domain.nodes.oldname, 
                           self.domain.cells.volume, self.domain.nodes.cellid, 
                           self.domain.cells.center, self.domain.halos.centvol, self.domain.nodes.halonid, self.domain.nodes.periodicid,
                           self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, self.domain.faces.airDiamond, 
                           self.domain.nodes.lambda_x, self.domain.nodes.lambda_y, self.domain.nodes.number, self.domain.nodes.R_x, self.domain.nodes.R_y, 
                           self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3, self.domain.faces.param4, self.domain.cells.shift, 
                           self.localsize, self.domain.cells.loctoglob, self.var.BCdirichlet, self._data, self._row, self._col, 
                           self._matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
        elif self.dim == 3:
            get_triplet_3d(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.vertex,
                           self.domain.faces.halofid, self.domain.halos.halosext, self.domain.nodes.name, self.domain.nodes.oldname, 
                           self.domain.cells.volume, self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, 
                           self.domain.nodes.periodicid, self.domain.nodes.cellid, self.domain.cells.center, self.domain.halos.centvol, 
                           self.domain.nodes.halonid, self.domain.faces.airDiamond, self.domain.nodes.lambda_x, self.domain.nodes.lambda_y,
                           self.domain.nodes.lambda_z, self.domain.nodes.number, self.domain.nodes.R_x, self.domain.nodes.R_y, self.domain.nodes.R_z, 
                           self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3,self.domain.cells.shift, self.domain.cells.loctoglob, 
                           self.var.BCdirichlet, self._data, self._row, self._col, self._matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
        if self.solver == "spsolve":
            from scipy import sparse
            self.mat = sparse.csr_matrix((self._data, (self._row, self._col)))
        
        
        elif self.solver == "umfpack":
            from scipy import sparse
            self.mat = sparse.csr_matrix((self._data, (self._row, self._col)))
        
        
        elif self.solver == "petsc":
            from petsc4py import PETSc
            
            NNZ_loc = np.zeros(self.globalsize, dtype=np.int32)
            unique, counts = np.unique(np.asarray(self._row), return_counts=True)
            
            
            for i in range(self.domain.nbcells):
                NNZ_loc[self.domain.cells.loctoglob[i]] = counts[i]
                
            NNZ = self.comm.allreduce(NNZ_loc, op=MPI.SUM)#, root=0)
            self.comm.Barrier()
    
            mat = PETSc.Mat().create()
            mat.setSizes(self.globalsize)
            mat.setType("mpiaij")
            mat.setFromOptions()
            mat.setPreallocationNNZ(NNZ)
            mat.setOption(option=19, flag=0)
            
            for i in range(len(self._row)):
                mat.setValues(self._row[i], self._col[i], self._data[i], addv=True)
            
            mat.assemblyBegin(mat.AssemblyType.FINAL)
            mat.assemblyEnd(mat.AssemblyType.FINAL)
            
            # print(mat.view())
            # import sys; sys.exit()
                
            self.sol = PETSc.Vec()
            self.sol.create()
            self.sol.setSizes(self.globalsize)
            self.sol.setFromOptions()
            
            pc = self.ksp.getPC()
            pc.setFactorSolverType("mumps")
        
            self.opts["ksp_type"] = "fgmres"
            self.opts["ksp_rtol"] = 1.0e-10
            self.opts["pc_type"] = "lu"
            self.opts["mat_mumps_icntl_18"] = 3
            self.opts["mat_mumps_icntl_28"] = 2
            # self.opts["mat_mumps_use_omp_threads"] = 3
            
            
            self.ksp.setInitialGuessNonzero(1)
            self.ksp.setFromOptions()
            self.ksp.setOperators(mat)
            
            # for i in range(self.localsize):
            #     self.sol.setValues(self.domain.cells.loctoglob[i], self._func(self.domain.cells.center[i][0], 
            #                                                                    self.domain.cells.center[i][1], 
            #                                                                    self.domain.cells.center[i][2]) )
            
            # solution vector
            self.sol.assemblyBegin()
            self.sol.assemblyEnd() 
            
            self.rhs = PETSc.Vec()
            self.rhs.create()
            self.rhs.setSizes(self.globalsize)
            self.rhs.setFromOptions()
            
        elif self.solver == "mumps":
            
            self.ctx.set_distributed_assembled_rows_cols(self._row+1, self._col+1)
            self.ctx.set_distributed_assembled_values(self._data)
            self.ctx.set_icntl(18,3)
            # self.ctx.set_icntl(28,2)
            self.ctx.set_icntl(16,2)
            
            if self.comm.Get_rank() == 0:
                self.ctx.id.n = self.globalsize
                self.sol = self.rhs0_glob.copy()
                
            #Analyse 
            self.ctx.run(job=1)
            #Factorization Phase
            self.ctx.run(job=2)
            
            #Allocation size of rhs
            if self.comm.Get_rank() == 0:
                self.ctx.set_rhs(self.sol)
            else :
                self.sol = np.zeros(self.globalsize)
            
    def solve(self, rhs=None, comm=None, compute_grad=None):
        
        if self.solver == "spsolve":
            if rhs is None:
                  rhs = self.rhs0_glob
            else:
                assert len(rhs) == len(self.rhs0_glob), 'rhs must have global shape for umfpack'
                rhs = rhs + self.rhs0_glob
        
            from scipy.sparse.linalg import spsolve
            self.sol = spsolve(self.mat, rhs)
            
            self.var.cell = self.sol
        
        
        elif self.solver == "umfpack":
            if rhs is None:
                  rhs = self.rhs0_glob
            else:
                assert len(rhs) == len(self.rhs0_glob), 'rhs must have global shape for umfpack'
                rhs = rhs + self.rhs0_glob
            
            from scikits.umfpack import splu
            lu = splu(self.mat)
            self.sol = lu.solve(rhs)
            
            self.var.cell = self.sol
            
        
        elif self.solver == "mumps":
            if rhs is None:
                  rhs = self.rhs0_glob
            else:
                assert len(rhs) == len(self.rhs0_glob), 'rhs must have global shape for mumps'
                rhs = rhs + self.rhs0_glob
            
            #Allocation size of rhs
            if self.comm.Get_rank() == 0:
                self.sol = rhs.copy()
                self.ctx.set_rhs(self.sol)
            
            #Solution Phase
            self.ctx.run(job=3)
            
            if self.comm.Get_rank() == 0:
                #Convert solution for scattering
                convert_solution(self.sol, self.x1converted, self.domain.cells.tc, self.globalsize)
            self.comm.Scatterv([self.x1converted, self.sendcounts1, MPI.DOUBLE], self.var.cell, root = 0)
            # self.ctx.destroy()
            
            
        elif self.solver == "petsc":
            if rhs is None:
                rhs = self.rhs0_loc
            else:
                assert len(rhs) == len(self.rhs0_loc), 'rhs must have local shape for petsc'
                rhs = rhs + self.rhs0_loc
            
            for i in range(self.domain.nbcells):
                self.rhs.setValues(self.domain.cells.loctoglob[i], rhs[i])#, addv=True)
            
            self.rhs.assemblyBegin()
            self.rhs.assemblyEnd()
            
            self.ksp.solve(self.rhs, self.sol)
            
            # print(self.rhs.view())
            # print(self.sol.view())
            # import sys; sys.exit()
            
            self.sendcounts2 = np.array(self.comm.gather(len(self.sol.array), root=0))
            
            if self.comm.Get_rank() == 0:
                recvbuf = np.empty(sum(self.sendcounts2), dtype=np.double)
            else:
                recvbuf = None
            
            self.comm.Gatherv(sendbuf=self.sol.array, recvbuf=(recvbuf, self.sendcounts2), root=0)
            
            if self.comm.Get_rank() == 0:
                #Convert solution for scattering
                convert_solution(recvbuf, self.x1converted, self.domain.cells.tc, self.globalsize)
                
            self.comm.Scatterv([self.x1converted, self.sendcounts1, MPI.DOUBLE], self.var.cell, root = 0)
            # self.ksp.destroy()
        
        if compute_grad:
            self.compute_Sol_gradient()

    def destroy(self):
        
        if self.solver == "mumps":
            self.ctx.destroy()
            
        elif self.solver == "petsc":
            self.ksp.destroy()