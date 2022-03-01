from pyccel.decorators import stack_array, pure
from numpy import double, zeros, sqrt, fabs

@pure
def compute_K(cellid:'int[:,:]', namef:'int[:]', ghostcenterf:'float[:,:]', centerc:'float[:,:]', K:'float[:,:]', dim:'int'):
    
    nbfaces = len(cellid)
    for i in range(nbfaces):
        if namef[i] <= 4 and namef[i] != 0 :
            c_left = cellid[i][0]
            K[i][0:dim] = 0.5*(centerc[c_left][0:dim] + ghostcenterf[i][0:dim])
@pure
@stack_array('norm', 'snom')
#@njit
def create_info_2dfaces(cellid:'int[:,:]', nodeid:'int[:,:]', namen:'int[:]', vertex:'double[:,:]', 
                        centerc:'double[:,:]', nbfaces:'int', normalf:'double[:,:]', mesuref:'double[:]',
                        centerf:'double[:,:]', namef:'int[:]'):
    
    #from numpy import double, zeros, sqrt
    
    norm   = zeros(3, dtype=double)
    snorm  = zeros(3, dtype=double)
    
    
    #Faces aux bords (1,2,3,4), Faces à l'interieur 0    A VOIR !!!!!
    for i in range(nbfaces):
        if (cellid[i][1] == -1 and cellid[i][1] != -10):
           
            if namen[nodeid[i][0]] == namen[nodeid[i][1]]:
                namef[i] = namen[nodeid[i][0]]
          
            elif ((namen[nodeid[i][0]] == 3 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 3)):
                namef[i] = 3
            elif ((namen[nodeid[i][0]] == 4 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 4)):
                namef[i] = 4
                
            elif ((namen[nodeid[i][0]] == 33 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 33)):
                namef[i] = 33
            elif ((namen[nodeid[i][0]] == 44 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 44)):
                namef[i] = 44  
            
            else:
                namef[i] = 100
        
        norm[0] = vertex[nodeid[i][0]][1] - vertex[nodeid[i][1]][1]
        norm[1] = vertex[nodeid[i][1]][0] - vertex[nodeid[i][0]][0]
    
        centerf[i][:] = 0.5 * (vertex[nodeid[i][0]][0:3] + vertex[nodeid[i][1]][0:3])
    
        snorm[:] = centerc[cellid[i][0]][:] - centerf[i][:]
    
        if (snorm[0] * norm[0] + snorm[1] * norm[1]) > 0:
            normalf[i][:] = -1*norm[:]
        else:
            normalf[i][:] = norm[:]

        mesuref[i] = sqrt(normalf[i][0]**2 + normalf[i][1]**2)

@pure   
@stack_array('norm', 'snom', 'u', 'v')
#@njit
def create_info_3dfaces(cellid:'int[:,:]', nodeid:'int[:,:]', namen:'int[:]', vertex:'double[:,:]', 
                        centerc:'double[:,:]', nbfaces:'int', normalf:'double[:,:]', mesuref:'double[:]',
                        centerf:'double[:,:]', namef:'int[:]'):
    
    #from numpy import double, zeros, sqrt
    
    norm   = zeros(3, dtype=double)
    snorm  = zeros(3, dtype=double)
    u      = zeros(3, dtype=double)
    v      = zeros(3, dtype=double)
    
    for i in range(nbfaces):
        if (cellid[i][1] == -1 ):
            if namen[nodeid[i][0]] == namen[nodeid[i][1]] and namen[nodeid[i][0]] == namen[nodeid[i][2]] :
                namef[i] = namen[nodeid[i][0]]
                
           
            elif ((namen[nodeid[i][0]] == 5 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 5 and namen[nodeid[i][2]] !=0) or 
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 5)):
                    namef[i] = 5
            
            elif ((namen[nodeid[i][0]] == 6 and namen[nodeid[i][1]] !=0 and namen[nodeid[i][2]] != 0) or
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 6 and namen[nodeid[i][2]] != 0) or 
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 6)):
                namef[i] = 6
                
             
            elif ((namen[nodeid[i][0]] == 3 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 3 and namen[nodeid[i][2]] != 0) or 
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 3)):
                    namef[i] = 3
            
            elif ((namen[nodeid[i][0]] == 4 and namen[nodeid[i][1]] !=0 and namen[nodeid[i][2]] != 0) or
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 4 and namen[nodeid[i][2]] != 0) or 
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 4)):
                namef[i] = 4
            
            
            elif ((namen[nodeid[i][0]] == 55 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 55 and namen[nodeid[i][2]] != 0) or 
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 55)):
                    namef[i] = 55
            
            elif ((namen[nodeid[i][0]] == 66 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 66 and namen[nodeid[i][2]] != 0) or 
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 66)):
                namef[i] = 66
                
            elif ((namen[nodeid[i][0]] == 33 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 33 and namen[nodeid[i][2]] != 0) or 
                  (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 33)):
                    namef[i] = 33
            
            elif ((namen[nodeid[i][0]] == 44 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 44 and namen[nodeid[i][2]] != 0) or 
                (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 44)):
                namef[i] = 44
                
                
            else:
                namef[i] = 100
                
        u[:] = vertex[nodeid[i][1]][0:3]-vertex[nodeid[i][0]][0:3]
        v[:] = vertex[nodeid[i][2]][0:3]-vertex[nodeid[i][0]][0:3]
        
        norm[0] = 0.5*(u[1]*v[2] - u[2]*v[1])
        norm[1] = 0.5*(u[2]*v[0] - u[0]*v[2])
        norm[2] = 0.5*(u[0]*v[1] - u[1]*v[0])
    
        centerf[i][:] = 1./3 * (vertex[nodeid[i][0]][:3] + vertex[nodeid[i][1]][:3] + vertex[nodeid[i][2]][:3])
    
        snorm[:] = centerc[cellid[i][0]][:] - centerf[i][:]
    
        if (snorm[0] * norm[0] + snorm[1] * norm[1] + snorm[2] * norm[2]) > 0:
            normalf[i][:] = -1*norm[:]
        else:
            normalf[i][:] = norm[:]
    
        mesuref[i] = sqrt(normalf[i][0]**2 + normalf[i][1]**2 + normalf[i][2]**2)
        
@pure
#@njit
def Compute_2dcentervolumeOfCell(nodeid:'int[:,:]', vertex:'double[:,:]', nbelements:'int',
                                 center:'double[:,:]', volume:'double[:]'):
    
    #calcul du barycentre et volume
    for i in range(nbelements):
        s_1 = nodeid[i][0]
        s_2 = nodeid[i][1]
        s_3 = nodeid[i][2]

        x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]
        x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]
        x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]

        center[i][0] = 1./3 * (x_1 + x_2 + x_3); center[i][1] = 1./3*(y_1 + y_2 + y_3); center[i][2] =  0.
        volume[i] = (1./2) * abs((x_1-x_2)*(y_1-y_3)-(x_1-x_3)*(y_1-y_2))
      
        var1 = (x_2-x_1)*(y_3-y_1)-(y_2-y_1)*(x_3-x_1)
        if var1 < 0:
            nodeid[i][0] = s_1;   nodeid[i][1] = s_3; nodeid[i][2] = s_2

@pure
@stack_array('u', 'v','w')
#@njit(fastmath=True)
def Compute_3dcentervolumeOfCell(nodeid:'int[:,:]', vertex:'double[:,:]', nbelements:'int',
                                 center:'double[:,:]', volume:'double[:]'):
    
    #from numpy import zeros, fabs
    wedge = zeros(3)
    u = zeros(3)
    v = zeros(3)
    w = zeros(3)
    
    #calcul du barycentre et volume
    for i in range(nbelements):
        
        s_1 = nodeid[i][0]
        s_2 = nodeid[i][1]
        s_3 = nodeid[i][2]
        s_4 = nodeid[i][3]
        
        x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]; z_1 = vertex[s_1][2]
        x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]; z_2 = vertex[s_2][2]
        x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]; z_3 = vertex[s_3][2]
        x_4 = vertex[s_4][0]; y_4 = vertex[s_4][1]; z_4 = vertex[s_4][2]
        
        center[i][0] = 1./4*(x_1 + x_2 + x_3 + x_4) 
        center[i][1] = 1./4*(y_1 + y_2 + y_3 + y_4)
        center[i][2] = 1./4*(z_1 + z_2 + z_3 + z_4)
        
        u[:] = vertex[s_2][0:3]-vertex[s_1][0:3]
        v[:] = vertex[s_3][0:3]-vertex[s_1][0:3]
        w[:] = vertex[s_4][0:3]-vertex[s_1][0:3]
        
        wedge[0] = v[1]*w[2] - v[2]*w[1]
        wedge[1] = v[2]*w[0] - v[0]*w[2]
        wedge[2] = v[0]*w[1] - v[1]*w[0]
        
        volume[i] = 1./6*fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 
        
@pure
#@njit
def create_cellsOfFace(faceid:'int[:,:]', nbelements:'int', nbfaces:'int', cellid:'int[:,:]', dim:'int'):

    for i in range(nbelements):
        for j in range(dim+1):
            if cellid[faceid[i][j]][0] == -1 :
                cellid[faceid[i][j]][0] = i

            if cellid[faceid[i][j]][0] != i:
                cellid[faceid[i][j]][0] = cellid[faceid[i][j]][0]
                cellid[faceid[i][j]][1] = i

@pure
#@njit(fastmath=True)
def create_2dfaces(nodeidc:'int[:,:]', nbelements:'int', faces:'int[:,:]',
                   cellf:'int[:,:]'):
   
    #Create 2d faces
    k = 0
    for i in range(nbelements):
        faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]
        faces[k+1][0] = nodeidc[i][1]; faces[k+1][1] = nodeidc[i][2]
        faces[k+2][0] = nodeidc[i][2]; faces[k+2][1] = nodeidc[i][0]
        cellf[i][0] = k; cellf[i][1] = k+1; cellf[i][2] = k+2
        k = k+3

@pure
#@njit
def create_cell_faceid(nbelements:'int', oldTonewIndex:'int[:]', cellf:'int[:,:]', 
                       faceid:'int[:,:]', dim:'int'):
    
    for i in range(nbelements):
        for j in range(dim+1):
            faceid[i][j] = oldTonewIndex[cellf[i][j]]
        
@pure
#@njit(fastmath=True)
def create_3dfaces(nodeidc:'int[:,:]', nbelements:'int',faces:'int[:,:]',
                   cellf:'int[:,:]'):
    #Create 3d faces
    k = 0
    
    for i in range(nbelements):
        faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]; faces[k][2]   = nodeidc[i][2]
        faces[k+1][0] = nodeidc[i][2]; faces[k+1][1] = nodeidc[i][3]; faces[k+1][2] = nodeidc[i][0]
        faces[k+2][0] = nodeidc[i][0]; faces[k+2][1] = nodeidc[i][1]; faces[k+2][2] = nodeidc[i][3]
        faces[k+3][0] = nodeidc[i][3]; faces[k+3][1] = nodeidc[i][1]; faces[k+3][2] = nodeidc[i][2]
        cellf[i][0]  = k; cellf[i][1] = k+1; cellf[i][2] = k+2; cellf[i][3] = k+3
        k = k+4
    
@pure
@stack_array('ss', 'G', 'c')
def create_NormalFacesOfCell(centerc:'double[:,:]', centerf:'double[:,:]', faceid:'int[:,:]', normal:'double[:,:]',
                             nbelements:'int', nf:'double[:,:,:]', dim:'int'):
    
    #from numpy import zeros
    ss = zeros(3)
    G  = zeros(3)
    c  = zeros(3)
    
    #compute the outgoing normal faces for each cell
    for i in range(nbelements):
        G[:] = centerc[i][:]

        for j in range(dim+1):
            f = faceid[i][j]
            c[:] = centerf[f][:]

            if ((G[0]-c[0])*normal[f][0] + (G[1]-c[1])*normal[f][1] + (G[2]-c[2])*normal[f][2]) < 0.:
                ss[:] = normal[f][:]
            else:
                ss[:] = -1.0*normal[f][:]
                
            nf[i][j][:] = ss[:]

@stack_array('xy_1', 'xy_2', 'v_1', 'v_2', 'shift')
def face_gradient_info_2d(cellidf:'int[:,:]', nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', normalf:'float[:,:]', 
                          centerc:'float[:,:]',  centerh:'float[:,:]', halofid:'int[:]', vertexn:'float[:,:]', 
                          airDiamond:'float[:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', param4:'float[:]', 
                          f_1:'float[:,:]', f_2:'float[:,:]', f_3:'float[:,:]', f_4:'float[:,:]', shift:'float[:,:]', 
                          dim:'int'):
    
    nbface = len(cellidf)
    
    xy_1 = zeros(dim)
    xy_2 = zeros(dim)
    v_1  = zeros(dim)
    v_2  = zeros(dim)
    
    for i in range(nbface):
        

        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]       
        
        xy_1[:] = vertexn[i_1][0:dim]
        xy_2[:] = vertexn[i_2][0:dim]
        
        v_1[:] = centerc[c_left][0:dim]
        
        if namef[i] == 0:
            v_2[:] = centerc[c_right][0:dim]
        elif namef[i] == 11 or namef[i] == 22 :
            v_2[0] = centerc[c_right][0] + shift[c_right][0]
            v_2[1] = centerc[c_right][1] 
        elif namef[i] == 33 or namef[i] == 44:
            v_2[0] = centerc[c_right][0]
            v_2[1] = centerc[c_right][1] + shift[c_right][1]
        elif namef[i] == 10:
            v_2[:] = centerh[halofid[i]][0:dim]
        else :
            v_2[:] = centergf[i][0:dim]

        f_1[i][:] = v_1[:] - xy_1[:]
        f_2[i][:] = xy_2[:] - v_1[:]
        f_3[i][:] = v_2[:] - xy_2[:]
        f_4[i][:] = xy_1[:] - v_2[:]
        
        n1 = normalf[i][0]
        n2 = normalf[i][1]
        
        airDiamond[i] = 0.5 *((xy_2[0] - xy_1[0]) * (v_2[1]-v_1[1]) + (v_1[0]-v_2[0]) * (xy_2[1] - xy_1[1]))
        
        param1[i] = 1./(2.*airDiamond[i]) * ((f_1[i][1]+f_2[i][1])*n1 - (f_1[i][0]+f_2[i][0])*n2)
        param2[i] = 1./(2.*airDiamond[i]) * ((f_2[i][1]+f_3[i][1])*n1 - (f_2[i][0]+f_3[i][0])*n2)
        param3[i] = 1./(2.*airDiamond[i]) * ((f_3[i][1]+f_4[i][1])*n1 - (f_3[i][0]+f_4[i][0])*n2)
        param4[i] = 1./(2.*airDiamond[i]) * ((f_4[i][1]+f_1[i][1])*n1 - (f_4[i][0]+f_1[i][0])*n2)


#TODO periodic checked ok 
@stack_array('center', 'shift')          
def variables(centerc:'float[:,:]', cellidn:'int[:,:]', haloidn:'int[:,:]', periodicn:'int[:,:]', 
              vertexn:'float[:,:]', namen:'int[:]', 
              centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', centerh:'float[:,:]', nbproc:'int',  
              R_x:'float[:]', R_y:'float[:]', lambda_x:'float[:]', 
              lambda_y:'float[:]', number:'int[:]', shift:'float[:,:]'):
    
      nbnode = len(R_x)
        
      I_xx = zeros(nbnode)
      I_yy = zeros(nbnode)
      I_xy = zeros(nbnode)
      center = zeros(3)
   
      for i in range(nbnode):
        for j in range(cellidn[i][-1]):
            center[:] = centerc[cellidn[i][j]][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_xy[i] += (Rx * Ry)
            R_x[i] += Rx
            R_y[i] += Ry
            number[i] += 1
            
        #ghost boundary (old vertex names)
        # if vertexn[i][3] == 1 or vertexn[i][3] == 2 or vertexn[i][3] == 3 or vertexn[i][3] == 4:
        if centergn[i][0][2] != -1: 
            for j in range(len(centergn[i])):
                cell = int(centergn[i][j][-1])
                if cell != -1:
                    center[:] = centergn[i][j][0:3]
                    Rx = center[0] - vertexn[i][0]
                    Ry = center[1] - vertexn[i][1]
                    I_xx[i] += (Rx * Rx)
                    I_yy[i] += (Ry * Ry)
                    I_xy[i] += (Rx * Ry)
                    R_x[i] += Rx
                    R_y[i] += Ry
                    number[i] = number[i] + 1
        
        #periodic boundary old vertex names)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[0] = centerc[cell][0] + shift[cell][0]
                center[1] = centerc[cell][1]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] += 1
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] == 44:
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[0] = centerc[cell][0]
                center[1] = centerc[cell][1] + shift[cell][1]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] += 1
                    
        if  namen[i] == 10 :
            for j in range(len(halocentergn[i])):
                cell = int(halocentergn[i][j][-1])
                if cell != -1:
                    center[:] = halocentergn[i][j][0:3]
                    Rx = center[0] - vertexn[i][0]
                    Ry = center[1] - vertexn[i][1]
                    
                    I_xx[i] += (Rx * Rx)
                    I_yy[i] += (Ry * Ry)
                    I_xy[i] += (Rx * Ry)
                    R_x[i] += Rx
                    R_y[i] += Ry
                    number[i] = number[i] + 1
            
            # if haloidn[i][-1] > 0:
            for j in range(haloidn[i][-1]):
                cell = haloidn[i][j]
                center[:] = centerh[cell][0:3]
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] = number[i] + 1
        

        D = I_xx[i]*I_yy[i] - I_xy[i]*I_xy[i]
        lambda_x[i] = (I_xy[i]*R_y[i] - I_yy[i]*R_x[i]) / D
        lambda_y[i] = (I_xy[i]*R_x[i] - I_xx[i]*R_y[i]) / D


#TODO periodic checked ok 
@stack_array('center')  
def variables_3d(centerc:'float[:,:]', cellidn:'int[:,:]', haloidn:'int[:,:]', periodicn:'int[:,:]',  vertexn:'float[:,:]', namen:'int[:]', 
                 centergn:'float[:,:,:]', halocenterg:'float[:,:,:]', centerh:'float[:,:]', nbproc:'int',
                 R_x:'float[:]', R_y:'float[:]', R_z:'float[:]', lambda_x:'float[:]', lambda_y:'float[:]', lambda_z:'float[:]', 
                 number:'int[:]', shift:'float[:,:]'):
    
    nbnode = len(R_x)
     
    I_xx = zeros(nbnode)
    I_yy = zeros(nbnode)
    I_zz = zeros(nbnode)
    I_xy = zeros(nbnode)
    I_xz = zeros(nbnode)
    I_yz = zeros(nbnode)
    center = zeros(3)
  

    for i in range(nbnode):
        for j in range(cellidn[i][-1]):
            center[:] = centerc[cellidn[i][j]][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            Rz = center[2] - vertexn[i][2]
           
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_zz[i] += (Rz * Rz)
            I_xy[i] += (Rx * Ry)
            I_xz[i] += (Rx * Rz)
            I_yz[i] += (Ry * Rz)
           
            R_x[i] += Rx
            R_y[i] += Ry
            R_z[i] += Rz
           
            number[i] += 1

        if centergn[i][0][3] != -1: 
            for j in range(len(centergn[i])):
                cell = int(centergn[i][j][-1])
                if cell != -1:
                    center[:] = centergn[i][j][0:3]
                    Rx = center[0] - vertexn[i][0]
                    Ry = center[1] - vertexn[i][1]
                    Rz = center[2] - vertexn[i][2]
                   
                    I_xx[i] += (Rx * Rx)
                    I_yy[i] += (Ry * Ry)
                    I_zz[i] += (Rz * Rz)
                    I_xy[i] += (Rx * Ry)
                    I_xz[i] += (Rx * Rz)
                    I_yz[i] += (Ry * Rz)
                   
                    R_x[i] += Rx
                    R_y[i] += Ry
                    R_z[i] += Rz
                    number[i] = number[i] + 1
                
        #periodic boundary old vertex names)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[0] = centerc[cell][0] + shift[cell][0]
                center[1] = centerc[cell][1]
                center[2] = centerc[cell][2]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] == 44:
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[0] = centerc[cell][0]
                center[1] = centerc[cell][1] + shift[cell][1]
                center[2] = centerc[cell][2]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
                
        elif vertexn[i][3] == 55 or vertexn[i][3] == 66:
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[0] = centerc[cell][0]
                center[1] = centerc[cell][1] 
                center[2] = centerc[cell][2] + shift[cell][2]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
     
        if  namen[i] == 10 :
            for j in range(haloidn[i][-1]):
                cell = haloidn[i][j]
                center[:] = centerh[cell][0:3]
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
       
            # if namen[i] == 10:
            for j in range(len(halocenterg[i])):
                cell = int(halocenterg[i][j][-1])
                if cell != -1:
                    center[:] = halocenterg[i][j][0:3]
                    Rx = center[0] - vertexn[i][0]
                    Ry = center[1] - vertexn[i][1]
                    Rz = center[2] - vertexn[i][2]
                   
                    I_xx[i] += (Rx * Rx)
                    I_yy[i] += (Ry * Ry)
                    I_zz[i] += (Rz * Rz)
                    I_xy[i] += (Rx * Ry)
                    I_xz[i] += (Rx * Rz)
                    I_yz[i] += (Ry * Rz)
                   
                    R_x[i] += Rx
                    R_y[i] += Ry
                    R_z[i] += Rz
                    number[i] = number[i] + 1
     
        D = I_xx[i]*I_yy[i]*I_zz[i] + 2*I_xy[i]*I_xz[i]*I_yz[i] - I_xx[i]*I_yz[i]*I_yz[i] - I_yy[i]*I_xz[i]*I_xz[i] - I_zz[i]*I_xy[i]*I_xy[i]
       
        lambda_x[i] = ((I_yz[i]*I_yz[i] - I_yy[i]*I_zz[i])*R_x[i] + (I_xy[i]*I_zz[i] - I_xz[i]*I_yz[i])*R_y[i] + (I_xz[i]*I_yy[i] - I_xy[i]*I_yz[i])*R_z[i]) / D
        lambda_y[i] = ((I_xy[i]*I_zz[i] - I_xz[i]*I_yz[i])*R_x[i] + (I_xz[i]*I_xz[i] - I_xx[i]*I_zz[i])*R_y[i] + (I_yz[i]*I_xx[i] - I_xz[i]*I_xy[i])*R_z[i]) / D
        lambda_z[i] = ((I_xz[i]*I_yy[i] - I_xy[i]*I_yz[i])*R_x[i] + (I_yz[i]*I_xx[i] - I_xz[i]*I_xy[i])*R_y[i] + (I_xy[i]*I_xy[i] - I_xx[i]*I_yy[i])*R_z[i]) / D
