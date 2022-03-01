#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:39:55 2022

@author: kissami
"""

from manapy.ast.core import Variable

from manapy.models.SWModel.tools import (update_SW, time_step_SW, explicitscheme_convective_SW, 
                                         term_source_srnh_SW, term_friction_SW, term_coriolis_SW)

from manapy.fvm.pyccel_fvm import explicitscheme_dissipative

from mpi4py import MPI

class ShallowWaterModel():
    
    variables=["h", "hu", "hv", "hc", "Z"]
    varbs = {}
    def __init__(self, domain=None, terms=None, parameters=None, scheme=None, order=None, comm=None, *args, **kwargs):
        if domain is None:
            raise ValueError("domain must be given")
        if order is None:
            order = 1
        if comm is None:
            comm = MPI.COMM_WORLD
            
        if terms is None:
            terms = ['source', 'dissipation', 'coriolis', 'friction', "convective"]
        else:
            terms.extend(["convective"])
            
        self.domain = domain
        self.terms = terms
        self.parameters = parameters
        self.grav = 9.81
        self.order = order
        self.comm = comm
        self.Dz = 0.
        
        if "dissipation" in self.terms:
            if "Dx" not in self.parameters:
                raise ValueError("Dx number must be given")
            else:
                self.Dx  = self.parameters["Dx"]
            if "Dy" not in self.parameters:
                raise ValueError("Dy number must be given") 
            self.Dy  = self.parameters["Dy"]
        else:  
            self.Dx  = 0.
            self.Dy  = 0.
            
        if "friction" in self.terms:
            if "Manning" not in self.parameters:
                raise ValueError("Manning number must be given")
            self.Mann  = self.parameters["Manning"]
        else:  
            self.Mann  = 0.
            
        
        if "coriolis" in self.terms:
            if "CoriolisForce" not in self.parameters:
                raise ValueError("Coriolis Force number must be given")
            else:
                self.fc  = self.parameters["Manning"]
        else:  
            self.fc  = 0.
        
        for var in self.variables:
            self.varbs[var] = Variable(domain=self.domain, terms=self.terms, parameters=self.parameters, name=var)
        
        self.h  = self.varbs['h']
        self.hu = self.varbs['hu']
        self.hv = self.varbs['hv']
        self.hc = self.varbs['hc']
        self.Z  = self.varbs['Z']
                
    def update_values(self):
        for var in self.varbs.values():
            var.update_values()
        
    def update_explicit_convective(self):
        
        if self.order == 2:
            self.h.compute_cell_gradient()
            self.hc.compute_cell_gradient()
            
        explicitscheme_convective_SW(self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective, self.Z.convective, 
                                     self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                                     self.h.ghost, self.hu.ghost, self.hv.ghost, self.hc.ghost, self.Z.ghost, self.h.halo, 
                                     self.hu.halo, self.hv.halo, self.hc.halo, self.Z.halo,
                                     self.h.gradcellx, self.h.gradcelly, self.h.gradhalocellx, self.h.gradhalocelly,
                                     self.hc.gradcellx, self.hc.gradcelly, self.hc.gradhalocellx, 
                                     self.hc.gradhalocelly, self.hc.psi, self.hc.psihalo, 
                                     self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                     self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.mesure, 
                                     self.domain.faces.normal, self.domain.faces.halofid,  
                                     self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, self.order)
    def update_explicit_dissipative(self):
    
        self.hc.compute_face_gradient()
        explicitscheme_dissipative(self.hc.gradfacex, self.hc.gradfacey, self.hc.gradfacez, self.domain.faces.cellid, 
                                   self.domain.faces.normal, self.domain.faces.name, self.hc.dissipation, self.Dx, self.Dy, self.Dz)
        
    def update_term_source(self):
        term_source_srnh_SW(self.h.source, self.hu.source, self.hv.source,self.hc.source, self.Z.source,
                            self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                            self.h.ghost, self.hu.ghost, self.hv.ghost, self.hc.ghost, self.Z.ghost,
                            self.h.halo, self.hu.halo, self.hv.halo, self.hc.halo, self.Z.halo,
                            self.h.gradcellx, self.h.gradcelly, self.hc.psi, self.h.gradhalocellx, self.h.gradhalocelly, 
                            self.hc.psihalo, 
                            self.domain.cells.nodeid, self.domain.cells.faceid, self.domain.cells.cellfid, self.domain.faces.cellid,
                            self.domain.cells.center, self.domain.cells.nf, 
                            self.domain.faces.name, self.domain.faces.center, self.domain.halos.centvol,
                            self.domain.nodes.vertex, self.domain.faces.halofid, self.order)
    
    
    def update_term_friction(self):
        term_friction_SW(self.h.cell, self.hu.cell, self.hv.cell, self.grav, self.Mann, self.time_step) 
        self.hu.update_values()
        self.hv.update_values()
    
    
    def update_term_coriolis(self):
        term_coriolis_SW(self.hu.cell, self.hv.cell, self.hu.coriolis, self.hv.coriolis, self.fc)
        
    def update_time_step(self, cfl=None):
       
        ######calculation of the time step
        dt_c = time_step_SW(self.h.cell, self.hu.cell, self.hv.cell, cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                            self.domain.cells.volume, self.domain.cells.faceid, self.Dx, self.Dy)
        
        self.time_step = self.comm.allreduce(dt_c, MPI.MIN)
    
    def update_solution(self):
        
        #update friction term
        self.update_term_friction() 
        
        #update the convective flux using explicit scheme
        self.update_explicit_convective()
        
        #update the dissipative flux 
        self.update_explicit_dissipative()
    
        #update the source using explicit scheme
        self.update_term_source()
        
        #update coriolis forces
        self.update_term_coriolis()
        
        update_SW(self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell ,
                  self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective, self.Z.convective,
                  self. h.source, self.hu.source, self.hv.source, self.hc.source, self.Z.source,
                  self.hc.dissipation, self.hu.coriolis, self.hv.coriolis,
                  0., 0., self.time_step, self.domain.cells.volume)
        
    