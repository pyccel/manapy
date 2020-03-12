# coding: utf-8
import meshio
import numpy as np
from mgmetis import metis
import os
import timeit
from collections import OrderedDict

__all__ = ['MeshPart']

def MeshPart(size, filename):
    if size > 1:
        ParaMesh(size, filename)
    else:
        SeqMesh(filename)

def SeqMesh(filename):
    def load_gmsh_mesh(filename):
        mesh = meshio.gmsh.read(filename)
        return mesh
    
    def create_cell_nodeid(mesh):
        cell_nodeid = []
    

        for i,j in mesh.cells.items():
            if i == "triangle":
                for k in range(len(j)):
                    cell_nodeid.append(list(j[k]))
                    cell_nodeid[k].sort()
        return cell_nodeid
    
    def define_ghost_node(mesh, nodes):
         ghost_nodes = [0]*len(nodes)
         
         for i,j in mesh.cell_data.items():
            if i=="line":
                x = j.get('gmsh:physical')
        
         #print(x)
        
         for i,j in mesh.cells.items():
            if i == "line":
                for k in range(len(j)):
                    for l in range(2):
                        if x[k] > 2:
                            ghost_nodes[j[k][l]] = int(x[k])
         for i,j in mesh.cells.items():
             if i == "line":
                 for k in range(len(j)):
                     for l in range(2):
                         if x[k] <= 2:
                             ghost_nodes[j[k][l]] = int(x[k])
         
                
         return ghost_nodes
    
    def create_nodes(mesh):
        nodes = []
        nodes = mesh.points
        return nodes

   
    
    start = timeit.default_timer()
    
    #load mesh
    mesh = load_gmsh_mesh(filename)
     
    #coordinates x, y of each node
    nodes = create_nodes(mesh)
    #nodes of each cell
    cell_nodeid = create_cell_nodeid(mesh)
    
    ghost_nodes =  define_ghost_node(mesh, nodes)
    
    
    if os.path.exists("mesh"+str(0)+".txt"):
        os.remove("mesh"+str(0)+".txt")
        #else:
        #    print("Can not delete the file as it doesn't exists")
    
    with open("mesh"+str(0)+".txt", "a") as text_file:
        text_file.write("Elements\n")
        np.savetxt(text_file, cell_nodeid, fmt='%u')
        text_file.write("EndElements\n")


    with open("mesh"+str(0)+".txt", "a") as text_file:
        text_file.write("Nodes\n")
        for i in range(len(nodes)):
            for j in range(3):
                text_file.write(str(nodes[i][j])+str(" "))
            text_file.write(str(ghost_nodes[i]))
            text_file.write("\n")#, fmt='%u')
        text_file.write("EndNodes\n")
      
#        
    stop = timeit.default_timer()
    
    print('Global Execution Time: ', stop - start)  

def ParaMesh(size, filename):
    
    def load_gmsh_mesh(filename):
        mesh = meshio.gmsh.read(filename)
        return mesh
    
    def create_cell_nodeid(mesh):
        cell_nodeid = []
    
        for i,j in mesh.cells.items():
            if i == "triangle":
                for k in range(len(j)):
                    cell_nodeid.append(list(j[k]))
                    cell_nodeid[k].sort()
        return cell_nodeid
    
    def define_ghost_node(mesh, nodes):
         ghost_nodes = [0]*len(nodes)
         
         for i,j in mesh.cell_data.items():
            if i=="line":
                x = j.get('gmsh:physical')
        
         #print(x)
        
         for i,j in mesh.cells.items():
            if i == "line":
                for k in range(len(j)):
                    for l in range(2):
                        if x[k] > 2:
                            ghost_nodes[j[k][l]] = x[k]
         for i,j in mesh.cells.items():
             if i == "line":
                 for k in range(len(j)):
                     for l in range(2):
                         if x[k] <= 2:
                             ghost_nodes[j[k][l]] = x[k]
                      
         return ghost_nodes

    
    start = timeit.default_timer()
    
    #load mesh
    mesh = load_gmsh_mesh(filename)
    

     
    #coordinates x, y of each node
    nodes = mesh.points#create_nodes(mesh.points)
    #nodes of each cell
    cell_nodeid = create_cell_nodeid(mesh)
    
    
    cell_nodeiddict = {tuple(cell_nodeid[0]): 0}
    for i in range(1,len(cell_nodeid)):
        cell_nodeiddict[tuple(cell_nodeid[i])] = i
    
    #ghost nodes
    ghost_nodes = define_ghost_node(mesh, nodes)
    
    stopmesh = timeit.default_timer()
    print("Reading mesh", stopmesh-start)

    nbelements = len(cell_nodeid)
    nbnodes = len(nodes)

    print("Number of Cells : ", nbelements)
    print("Number of Nodes : ", nbnodes)

    
    #Partitioning mesh
    if size > 1:
        objval, epart, npart = metis.part_mesh_dual(size, cell_nodeid)
    
    stopmetis = timeit.default_timer()
    print("METIS partitionning in ",size, "partitions", stopmetis - stopmesh)
    
    
    node_parts= OrderedDict()
    cell_parts= OrderedDict()
    node_part = [[]  for i in range(size)]
    cell_part = [[]  for i in range(size)]
    #
    GlobNodeToLoc = OrderedDict()
    LocNodeToGlob = OrderedDict()
    GlobCellToLoc = OrderedDict()
    
    neighsub  = [[]  for i in range(size)]
    halo_cellid  = [[]  for i in range(size)]
    #
    npart = [[] for i in range(nbnodes)]
    cpart = [[] for i in range(nbelements)]
    
    for i in range(nbelements):
        for j in range(3):
            if epart[i] not in npart[cell_nodeid[i][j]]:
                npart[cell_nodeid[i][j]].append(epart[i])
     
    for i in range(nbelements):
        for j in range(3):      
            for k in range(len(npart[cell_nodeid[i][j]])):
                if npart[cell_nodeid[i][j]][k] not in cpart[i]:
                    cpart[i].append(npart[cell_nodeid[i][j]][k])
        cpart[i].sort()
        

    
    
    #Create dict of nodes/cells for each partition
    for i in range(nbelements):  
        for j in range(3):
            k = cell_nodeid[i][j]
            node_parts[epart[i],k] = [nodes[k][0], nodes[k][1], nodes[k][2], ghost_nodes[k]]
            cell_parts[epart[i], i ] = cell_nodeid[i]
     
    
    #Create list of nodes/cells for each partition and local to global indexation 
    for i,j in node_parts.items():
        node_part[i[0]].append(j)
        GlobNodeToLoc[i[0],i[1]] =  len(node_part[i[0]])-1
        LocNodeToGlob[i[0], len(node_part[i[0]])-1] = i[1]
        if (len(npart[i[1]]) > 1):
            for l in range(len(npart[i[1]])):
                if (npart[i[1]][l] not in neighsub[i[0]] and npart[i[1]][l]!=i[0]):
                    neighsub[i[0]].append(npart[i[1]][l])
                    neighsub[i[0]].sort()
    
    for i,j in cell_parts.items():
        cell_part[i[0]].append(j)
        GlobCellToLoc[i[0], len(cell_part[i[0]])-1] = i[1]
        #GlobCellToLoc[i[0],i[1]] = len(cell_part[i[0]])
        
    
    
    stopstruc = timeit.default_timer()
    print("Create local structure for each proc", stopstruc - stopmetis)
    
    ##
    
    for i in range(size):
        for j in cell_part[i]:
            if ((len(npart[j[0]]) + len(npart[j[1]]) + len(npart[j[2]])) > 3 ):
                    halo_cellid[i].append(j)
             
    #
#                    
    haloint  = OrderedDict()#
    haloext  = OrderedDict()
    for i in range(size):
        for j in halo_cellid[i]:
           m =  cell_nodeiddict.get(tuple(j))
           for k in range(len(cpart[m])):
               if (i != cpart[m][k]):
                   haloint.setdefault((i,cpart[m][k]), [] ).append(m)
                   haloext.setdefault((cpart[m][k],i), []).append(m)#haloint[(i,cpart[m][k])])

    for i in range(size):
        for j in range(len(cell_part[i])):
                cell_part[i][j] = [GlobNodeToLoc[i,cell_part[i][j][0]], 
                          GlobNodeToLoc[i,cell_part[i][j][1]], 
                          GlobNodeToLoc[i,cell_part[i][j][2]]]
    #
    stophalo = timeit.default_timer()         
    print("Creating halo structure", stophalo - stopstruc)
    #
    
    for i in range(size):
        if os.path.exists("mesh"+str(i)+".txt"):
            os.remove("mesh"+str(i)+".txt")
      
    for i in range(size):
        with open("mesh"+str(i)+".txt", "a") as text_file:
            text_file.write("Elements\n")
            np.savetxt(text_file, cell_part[i], fmt='%u')
            text_file.write("EndElements\n")
            text_file.write("Nodes\n")
            np.savetxt(text_file, node_part[i])
            text_file.write("EndNodes\n")
            text_file.write("HalosInt\n")
            for j in range(len(neighsub[i])):
                for k in range(len(haloint[(i,neighsub[i][j])])):
                    text_file.write(str(haloint[(i,neighsub[i][j])][k]))
                    text_file.write("\n")
            text_file.write("EndHalosInt\n")
            text_file.write("HalosExt\n")
            for j in range(len(neighsub[i])):
                for k in range(len(haloext[(i,neighsub[i][j])])):
                    text_file.write(str(cell_nodeid[haloext[(i,neighsub[i][j])][k]][0])+" "+
                                    str(cell_nodeid[haloext[(i,neighsub[i][j])][k]][1])+" "+
                                    str(cell_nodeid[haloext[(i,neighsub[i][j])][k]][2]))
                    text_file.write("\n")
            text_file.write("EndHalosExt\n")
            text_file.write("GlobalCellToLocal\n")
            for j in range(len(cell_part[i])):
                text_file.write(str(GlobCellToLoc[i,j]))
                text_file.write("\n")
            text_file.write("EndGlobalCellToLocal\n")
            text_file.write("LocalNodeToGlobal\n")
            for j in range(len(node_part[i])):
                text_file.write(str(LocNodeToGlob[i,j]))
                text_file.write("\n")
            text_file.write("EndLocalNodeToGlobal\n") 
            text_file.write("Neigh\n")
            for j in range(len(neighsub[i])):
                text_file.write(str(neighsub[i][j])+ " ")
            text_file.write("\n")
            for j in neighsub[i]:
                text_file.write(str(len(haloint[(i,j)]))+ " ")
            text_file.write("\n")
            text_file.write("EndNeigh\n")
    

    
    stopfile = timeit.default_timer() 
    print("save structures in files", stopfile - stophalo)
    
    stop = timeit.default_timer()
    
    print('Global Execution Time: ', stop - start)  

if __name__ == "__main__":
    #print("Entrer le fichier du maillage (.gmsh)")
    filename = "mesh.msh"#input()
    print("Entrer le nombre de partition souhaitee")
    size = int(input())
    MeshPart(size, filename)
