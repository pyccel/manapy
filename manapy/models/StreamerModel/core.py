#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:30:45 2022

@author: kissami
"""

from manapy.ast import Variable, LinearSystem

from manapy.models.StreamerModel.tools import (explicitscheme_dissipative_streamer, explicitscheme_source_streamer,
                                               update_streamer_flux, time_step_streamer, update_rhs_glob, update_rhs_loc,
                                               compute_el_field, compute_velocity)

from manapy.fvm.pyccel_fvm import (explicitscheme_convective_2d, 
                                   explicitscheme_convective_3d)
from numpy import zeros

from mpi4py import MPI

class StreamerModel():
    
    variables=["ne", "ni", "u", "v", "w",  "Ex", "Ey", "Ez", "P" ] 
    varbs = {}
    def __init__(self, domain=None, terms=None,  order=None, comm=None, *args, **kwargs):
        
        if domain is None:
            raise ValueError("domain must be given")
        if order is None:
            order = 1
        if comm is None:
            comm = MPI.COMM_WORLD
           
        if terms is None:
            terms = ['convective', 'source', 'dissipation']
        else:
            terms.extend(["convective"])
            
        self.domain = domain
        self.terms = terms
        self.order = order
        self.comm = comm
        self.dim  = self.domain.dim
        
        
        for var in self.variables:
            self.varbs[var] = Variable(domain=self.domain, terms=self.terms, name=var)
        
        if self.dim == 2:
            self.explicitscheme_convective = explicitscheme_convective_2d
        elif self.dim == 3:
            self.explicitscheme_convective = explicitscheme_convective_3d
            
            
        self.ne  = self.varbs['ne']
        self.ni = self.varbs['ni']
        self.u = self.varbs['u']
        self.v = self.varbs['v']
        self.w = self.varbs['w']
        self.Ex = self.varbs['Ex']
        self.Ey = self.varbs['Ey']
        self.Ez = self.varbs['Ez']
        self.P = self.varbs['P']
        
    def initiate_LS(self, solver=None, debug=None):
        if solver is None:
            solver = "spsolve"
            
        self.solver = solver
        
        self.L = LinearSystem(domain=self.domain, var= self.P, solver=self.solver, debug=debug)
        self.L.assembly()
       
        if self.solver == "petsc":
            self.update_rhs = update_rhs_loc
            self.rhs_updated = zeros(self.L.localsize)
        else:
            self.update_rhs = update_rhs_glob
            self.rhs_updated = zeros(self.L.globalsize)

        
    def update_values(self):
        for var in self.varbs.values():
            var.update_values()
        
    def update_explicit_convective(self):
        
        if self.order == 2:
            self.ne.compute_cell_gradient()
        
        self.explicitscheme_convective(self.ne.convective, self.ne.cell, self.ne.ghost, self.ne.halo,
                                       self.u.face, self.v.face, self.w.face, self.ne.gradcellx, self.ne.gradcelly, self.ne.gradcellz,
                                       self.ne.gradhalocellx, self.ne.gradhalocelly, self.ne.gradhalocellz, 
                                       self.ne.psi, self.ne.psihalo, self.domain.cells.center, self.domain.faces.center, 
                                       self.domain.halos.centvol, self.domain.faces.ghostcenter, self.domain.faces.cellid,
                                       self.domain.faces.mesure, self.domain.faces.normal, self.domain.faces.halofid,  
                                       self.domain.faces.name, self.domain.innerfaces, self.domain.halofaces,
                                       self.domain.boundaryfaces, self.domain.periodicboundaryfaces, self.domain.cells.shift, 
                                       self.order)
    
    def update_explicit_diffusion(self):
    
        self.ne.compute_face_gradient()
        explicitscheme_dissipative_streamer(self.u.face, self.v.face, self.w.face, self.Ex.face, self.Ey.face, self.Ez.face, self.ne.gradfacex, 
                                            self.ne.gradfacey, self.ne.gradfacez, self.domain.faces.cellid, self.domain.faces.normal, 
                                            self.domain.faces.name, self.ne.dissipation)
    
    def update_term_source(self, branching=0):
        
        explicitscheme_source_streamer(self.ne.cell, self.u.cell, self.v.cell, self.w.cell, self.Ex.cell, self.Ey.cell, 
                                       self.Ez.cell, self.ne.source, self.ni.source, self.domain.cells.center, branching)
    
    def solve_linearSystem(self):
       
        self.update_rhs(self.ne.cell, self.ni.cell, self.domain.cells.loctoglob, self.rhs_updated)
        
        if self.solver !="petsc":
            rhs = self.comm.reduce(self.rhs_updated, op=MPI.SUM, root=0)
        else:
            rhs = self.rhs_updated
        
        self.L.solve(compute_grad=True, rhs=rhs)    
        
        compute_el_field(self.P.gradfacex, self.P.gradfacey, self.P.gradfacez, self.Ex.face, self.Ey.face, self.Ez.face)
        
        compute_velocity(self.Ex.face, self.Ey.face, self.Ez.face, self.u.face, self.v.face, self.w.face, self.Ex.cell, self.Ey.cell,
                         self.Ez.cell, self.u.cell, self.v.cell, self.w.cell, self.domain.cells.faceid, self.dim)
        
        
    
    def update_time_step(self, cfl):
        ######calculation of the time step
        dt_c = time_step_streamer(self.u.cell, self.v.cell, self.w.cell, self.Ex.cell, self.Ey.cell, 
                                  self.Ez.cell, cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                                  self.domain.cells.volume, self.domain.cells.faceid, self.dim)
        
        self.time_step = self.comm.allreduce(dt_c, MPI.MIN)
    
    def update_solution(self):
        
        #update source term
        self.update_term_source()
        
        #update the convectiveidus using explicit scheme
        self.update_explicit_convective()
       
        #update the dissipative term
        self.update_explicit_diffusion()
        
        update_streamer_flux(self.ne.cell, self.ni.cell, self.ne.convective, self.ni.convective, self.ne.dissipation, self.ni.dissipation,
                             self.ne.source, self.ni.source, self.time_step, self.domain.cells.volume)
    
    
    def destroy(self):
        self.L.destroy()
      
        del self.__dict__
        
        for var in self.varbs.values():
            del var.__dict__