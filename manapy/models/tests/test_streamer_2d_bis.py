from manapy.ddm import Domain, readmesh

from manapy.models.StreamerModel import StreamerModel, initialisation_streamer
from manapy.ast import Variable

import timeit
import os
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')

#File name
filename = "rectangle.msh"
filename = os.path.join(MESH_DIR, filename)
    

dim = 3
readmesh(filename, dim=dim, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)

faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbcells = domain.nbcells

if RANK == 0: print("Start Computation")
cfl = 0.4
time = 0
tfinal = 5.25e-8
miter = 0
niter = 1
Pinit = 25000.
order = 2

boundaries = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }
values = {"in" : Pinit,
          "out": 0.,
          }


Model = StreamerModel(domain, terms=['convective', 'source', 'dissipation'], order=order)

Model.P = Variable(domain=domain, BC=boundaries, values=values)


ne = Model.ne
ni = Model.ni
u  = Model.u
v  = Model.v
w  = Model.w
Ex = Model.Ex
Ey = Model.Ey
Ez = Model.Ez
P  = Model.P

# Initialization
initialisation_streamer(ne.cell, ni.cell, u.cell, v.cell, Ex.cell, Ey.cell, P.cell, cells.center, Pinit=Pinit)

# Initiate the linear system (assembling matrix and rhs, ...) 
Model.initiate_LS(solver="mumps")

if RANK == 0: print("Start loop")

#loop over time
while time < tfinal:
    
    #Solving the linear system
    Model.solve_linearSystem()
    
    #update time step
    Model.update_time_step(cfl=cfl)
    time = time + Model.time_step
   
    #saving 50 vtk file
    tot = int(tfinal/Model.time_step/50)
    
    #update solution   
    Model.update_solution()
    
    #save vtk files for the solution
    if niter== 1 or niter%tot == 0:
        #interpolate solution on node
        ne.interpolate_celltonode()
        
        #save paraview result
        domain.save_on_node(Model.time_step, time, niter, miter, value=ne.node)
        miter += 1

    niter += 1


stop = timeit.default_timer()

Model.destroy()
