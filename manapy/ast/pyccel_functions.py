from pyccel.decorators import stack_array, inline

@inline
def convert_solution(x1:'float[:]', x1converted:'float[:]', tc:'int[:]', b0Size:'int'):
    #$ omp parallel for 
    for i in range(b0Size):
        x1converted[i] = x1[tc[i]]
    
@inline
def rhs_value_dirichlet_node(Pbordnode:'float[:]', nodes:'int[:]', value:'float[:]'):
    
    #$ omp parallel for 
    for i in nodes:
        Pbordnode[i] = value[i]
    
@inline       
def rhs_value_dirichlet_face(Pbordface:'float[:]', faces:'int[:]', value:'float[:]'):
    
    #$ omp parallel for 
    for i in faces:
        Pbordface[i] = value[i]

@inline
def ghost_value_slip(u_c:'float[:]', v_c:'float[:]', w_ghost:'float[:]', 
                     cellid:'int[:,:]', faces:'int[:]', normal:'float[:,:]', mesure:'float[:]'):
    
    from numpy import zeros
    s_n = zeros(3)
   
    #$ omp parallel for 
    for i in faces:
        
        u_i = u_c[cellid[i][0]]
        v_i = v_c[cellid[i][0]]
        
        s_n[:] = normal[i][:] / mesure[i]
        u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
        
        w_ghost[i] = u_c[cellid[i][0]] * u_g

@inline
def ghost_value_nonslip(w_c:'float[:]', w_ghost:'float[:]', cellid:'int[:,:]', faces:'int[:]'):
    
    #$ omp parallel for 
    for i in faces:
        w_ghost[i]  = -1*w_c[cellid[i][0]]

@inline
def ghost_value_neumann(w_c:'float[:]', w_ghost:'float[:]', cellid:'int[:,:]', faces:'int[:]'):
    
    #$ omp parallel for 
    for i in faces:
        w_ghost[i]  = w_c[cellid[i][0]]
     
@inline
def ghost_value_dirichlet(value:'float[:]', w_ghost:'float[:]', cellid:'int[:,:]', faces:'int[:]'):
    
    #$ omp parallel for 
    for i in faces:
        w_ghost[i]  = value[i]
    

def haloghost_value_neumann(w_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                            BCindex: 'int', halonodes:'int[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellhalo  = int(haloghostcenter[i][j][-3])
                    cellghost = int(haloghostcenter[i][j][-1])
    
                    w_haloghost[cellghost]   = w_halo[cellhalo]

def haloghost_value_dirichlet(value:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                              BCindex: 'int', halonodes:'int[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellghost = int(haloghostcenter[i][j][-1])
                    w_haloghost[cellghost]   = value[cellghost]

def haloghost_value_nonslip(w_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                           BCindex: 'int', halonodes:'int[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellghost = int(haloghostcenter[i][j][-1])
                    w_haloghost[cellghost]   = -1*w_halo[cellghost]


def haloghost_value_slip(u_halo:'float[:]', v_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                         BCindex: 'int', halonodes:'int[:]', haloghostfaceinfo:'float[:,:,:]'):
    
    from numpy import sqrt, zeros

    s_n = zeros(2)
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellghost = int(haloghostcenter[i][j][-1])

                    u_i = u_halo[cellghost]
                    v_i = v_halo[cellghost]
                    
                    mesure = sqrt(haloghostfaceinfo[i][j][2]**2 + haloghostfaceinfo[i][j][3]**2)# + haloghostfaceinfo[i][5]**2)
                    
                    s_n[0] = haloghostfaceinfo[i][j][2] / mesure
                    s_n[1] = haloghostfaceinfo[i][j][3] / mesure
                    
                    u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
                        
                    w_haloghost[i] = u_halo[cellghost] * u_g

@stack_array('center', 'shift')          
def cell_gradient_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                     centerc:'float[:,:]', cellnid:'int[:,:]', halonid:'int[:,:]',
                     nodecid:'int[:,:]', periodicn:'int[:,:]', periodic:'int[:,:]', namen:'int[:]', centerg:'float[:,:,:]', 
                     halocenterg:'float[:,:,:]', vertexn:'float[:,:]', centerh:'float[:,:]', shift:'float[:,:]',
                     nbproc:'int', w_x:'float[:]', w_y:'float[:]', w_z:'float[:]'):
                          
    
    from numpy import zeros
    center = zeros(3)
    nbelement = len(w_c)
    
    for i in range(nbelement):
        i_xx  = 0.;  i_yy  = 0.; i_xy = 0.
        j_xw = 0.;  j_yw = 0.

        for j in range(cellnid[i][-1]):
            cell = cellnid[i][j]
            j_x = centerc[cell][0] - centerc[i][0]
            j_y = centerc[cell][1] - centerc[i][1]
            i_xx += j_x*j_x
            i_yy += j_y*j_y
            i_xy += (j_x * j_y)

            j_xw += (j_x * (w_c[cell] - w_c[i] ))
            j_yw += (j_y * (w_c[cell] - w_c[i] ))
           
        for k in range(3):
            nod = nodecid[i][k]
            if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
                for j in range(periodic[nod][-1]):
                    cell = int(periodic[nod][j])
                    center[:] = centerc[cell][0:3]
                    j_x = center[0] + shift[cell][0] - centerc[i][0]
                    j_y = center[1] - centerc[i][1]
                    
                    i_xx += j_x*j_x
                    i_yy += j_y*j_y
                    i_xy += (j_x * j_y)
                    
                    j_xw += (j_x * (w_c[cell] - w_c[i] ))
                    j_yw += (j_y * (w_c[cell] - w_c[i] ))
                    
            if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
                for j in range(periodic[nod][-1]):
                    cell = int(periodic[nod][j])
                    center[:] = centerc[cell][0:3]
                    j_x = center[0] - centerc[i][0]
                    j_y = center[1] + shift[cell][1] - centerc[i][1]
                    
                    i_xx += j_x*j_x
                    i_yy += j_y*j_y
                    i_xy += (j_x * j_y)
                    
                    j_xw += (j_x * (w_c[cell] - w_c[i] ))
                    j_yw += (j_y * (w_c[cell] - w_c[i] ))
                    

        for j in range(halonid[i][-1]):
            cell = halonid[i][j]
            j_x = centerh[cell][0] - centerc[i][0]
            j_y = centerh[cell][1] - centerc[i][1]
            
            i_xx += j_x*j_x
            i_yy += j_y*j_y
            i_xy += (j_x * j_y)
            
            j_xw += (j_x * (w_halo[cell]  - w_c[i] ))
            j_yw += (j_y * (w_halo[cell]  - w_c[i] ))
                
                
        for k in range(3):
            nod = nodecid[i][k]
            if vertexn[nod][3] <= 4:
                for j in range(len(centerg[nod])):
                    cell = int(centerg[nod][j][-1])
                    if cell != -1:
                        center[:] = centerg[nod][j][0:3]
                        j_x = center[0] - centerc[i][0]
                        j_y = center[1] - centerc[i][1]
                        
                        i_xx += j_x*j_x
                        i_yy += j_y*j_y
                        i_xy += (j_x * j_y)
                        j_xw += (j_x * (w_ghost[cell] - w_c[i] ))
                        j_yw += (j_y * (w_ghost[cell] - w_c[i] ))
                        
            for j in range(len(halocenterg[nod])):
                #-3 the index of global face
                cell = int(halocenterg[nod][j][-1])
                if cell != -1:
                    center[:] = halocenterg[nod][j][0:3]
                    j_x = center[0] - centerc[i][0]
                    j_y = center[1] - centerc[i][1]
                    
                    i_xx += j_x*j_x
                    i_yy += j_y*j_y
                    i_xy += (j_x * j_y)
    
                    j_xw += (j_x * (w_haloghost[cell] - w_c[i] ))
                    j_yw += (j_y * (w_haloghost[cell] - w_c[i] ))

        dia = i_xx * i_yy - i_xy*i_xy

        w_x[i]  = (i_yy * j_xw - i_xy * j_yw) / dia
        w_y[i]  = (i_xx * j_yw - i_xy * j_xw) / dia
        w_z[i]  = 0.

#TODO still does not work
@stack_array('center')    
def cell_gradient_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                     centerc:'float[:,:]', cellnid:'int[:,:]', halonid:'int[:,:]',
                     nodecid:'int[:,:]', periodicn:'int[:,:]', periodic:'int[:,:]', namen:'int[:]', centerg:'float[:,:,:]', 
                     halocenterg:'float[:,:,:]', vertexn:'float[:,:]', centerh:'float[:,:]', shift:'float[:,:]',
                     nbproc:'int', w_x:'float[:]', w_y:'float[:]', w_z:'float[:]'):
   
    from numpy import zeros
    
    nbelement = len(w_c)
    center = zeros(3)
    
    for i in range(nbelement):
        i_xx = 0.
        i_yy = 0.
        i_zz = 0.
        i_xy = 0.
        i_xz = 0.
        i_yz = 0.
        
        j_x = 0.
        j_y = 0.
        j_z = 0.

        for j in range(cellnid[i][-1]):
            cell = cellnid[i][j]
            jx = centerc[cell][0] - centerc[i][0]
            jy = centerc[cell][1] - centerc[i][1]
            jz = centerc[cell][2] - centerc[i][2]
            
            i_xx += jx*jx
            i_yy += jy*jy
            i_zz += jz*jz
            i_xy += (jx * jy)
            i_xz += (jx * jz)
            i_yz += (jy * jz)

            j_x += (jx * (w_c[cell] - w_c[i] ))
            j_y += (jy * (w_c[cell] - w_c[i] ))
            j_z += (jz * (w_c[cell] - w_c[i] ))
            
        for j in range(periodicn[i][-1]):
            cell = periodicn[i][j]
            center[:] = centerc[cell][0:3]
            j_x = center[0] + shift[cell][0] - centerc[i][0]
            j_y = center[1] + shift[cell][1] - centerc[i][1]
            j_z = center[2] + shift[cell][2] - centerc[i][2]
            
            i_xx += jx*jx
            i_yy += jy*jy
            i_zz += jz*jz
            i_xy += (jx * jy)
            i_xz += (jx * jz)
            i_yz += (jy * jz)

            j_x += (jx * (w_c[cell] - w_c[i] ))
            j_y += (jy * (w_c[cell] - w_c[i] ))
            j_z += (jz * (w_c[cell] - w_c[i] ))
      
        #if nbproc > 1:      
        for j in range(halonid[i][-1]):
            cell = halonid[i][j]
            jx = centerh[cell][0] - centerc[i][0]
            jy = centerh[cell][1] - centerc[i][1]
            jz = centerh[cell][2] - centerc[i][2]
            
            i_xx += jx*jx
            i_yy += jy*jy
            i_zz += jz*jz
            i_xy += (jx * jy)
            i_xz += (jx * jz)
            i_yz += (jy * jz)

            j_x += (jx * (w_halo[cell] - w_c[i] ))
            j_y += (jy * (w_halo[cell] - w_c[i] ))
            j_z += (jz * (w_halo[cell] - w_c[i] ))
        
        #TODO verify ghost center
        for k in range(4):
            nod = nodecid[i][k]
            if vertexn[nod][3] <= 6:
                for j in range(len(centerg[nod])):
                    cell = int(centerg[nod][j][-1])
                    if cell != -1:
                        center[:] = centerg[nod][j][0:3]
                        jx = center[0] - centerc[i][0]
                        jy = center[1] - centerc[i][1]
                        jz = center[2] - centerc[i][2]
                        
                        i_xx += jx*jx
                        i_yy += jy*jy
                        i_zz += jz*jz
                        i_xy += (jx * jy)
                        i_xz += (jx * jz)
                        i_yz += (jy * jz)
        
                        j_x += (jx * (w_ghost[cell] - w_c[i] ))
                        j_y += (jy * (w_ghost[cell] - w_c[i] ))
                        j_z += (jz * (w_ghost[cell] - w_c[i] ))
                        
            #if namen[nod] == 10 :
            for j in range(len(halocenterg[nod])):
                #-3 the index of global face
                cell = int(halocenterg[nod][j][-1])
                if cell != -1:
                    center[:] = halocenterg[nod][j][0:3]
                    jx = center[0] - centerc[i][0]
                    jy = center[1] - centerc[i][1]
                    jz = center[2] - centerc[i][2]
                    
                    i_xx += jx*jx
                    i_yy += jy*jy
                    i_zz += jz*jz
                    i_xy += (jx * jy)
                    i_xz += (jx * jz)
                    i_yz += (jy * jz)
    
                    j_x += (jx * (w_haloghost[cell] - w_c[i] ))
                    j_y += (jy * (w_haloghost[cell] - w_c[i] ))
                    j_z += (jz * (w_haloghost[cell] - w_c[i] ))

        
        dia = i_xx*i_yy*i_zz + 2.*i_xy*i_xz*i_yz - i_xx*i_yz**2 - i_yy*i_xz**2 - i_zz*i_xy**2

        w_x[i] = ((i_yy*i_zz - i_yz**2)*j_x   + (i_xz*i_yz - i_xy*i_zz)*j_y + (i_xy*i_yz - i_xz*i_yy)*j_z) / dia
        w_y[i] = ((i_xz*i_yz - i_xy*i_zz)*j_x + (i_xx*i_zz - i_xz**2)*j_y   + (i_xy*i_xz - i_yz*i_xx)*j_z) / dia
        w_z[i] = ((i_xy*i_yz - i_xz*i_yy)*j_x + (i_xy*i_xz - i_yz*i_xx)*j_y + (i_xx*i_yy - i_xy**2)*j_z  ) / dia


@stack_array('shift')
def face_gradient_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_node:'float[:]', cellidf:'int[:,:]', 
                     nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', halofid:'int[:]', centerc:'float[:,:]', 
                     centerh:'float[:,:]', vertexn:'float[:,:]', airDiamond:'float[:]', normalf:'float[:,:]',
                     f_1:'float[:,:]', f_2:'float[:,:]', f_3:'float[:,:]', f_4:'float[:,:]', shift:'float[:,:]', 
                     wx_face:'float[:]', wy_face:'float[:]', wz_face:'float[:]', innerfaces:'int[:]', halofaces:'int[:]',
                     dirichletfaces:'int[:]', neumann:'int[:]', periodicfaces:'int[:]'):

    for i in innerfaces:
       
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
            
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = w_node[i_1]
        vi2 = w_node[i_2]
        vv1 = w_c[c_left]
        vv2 = w_c[c_right]
        
        wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    for i in periodicfaces:
     
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
            
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = w_node[i_1]
        vi2 = w_node[i_2]
        vv1 = w_c[c_left]
        vv2 = w_c[c_right]
        
        wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    for i in halofaces:
       
        c_left = cellidf[i][0]
        c_right = halofid[i]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = w_node[i_1]
        vi2 = w_node[i_2]
        vv1 = w_c[c_left]
        vv2 = w_halo[c_right]
        
        wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    
    for i in dirichletfaces:
       
        c_left = cellidf[i][0]
        c_right = i
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = w_node[i_1]
        vi2 = w_node[i_2]
        vv1 = w_c[c_left]
        vv2 = w_ghost[c_right]

        wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    
    for i in neumann:
     
        c_left = cellidf[i][0]
        c_right = i
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = w_node[i_1]
        vi2 = w_node[i_2]
        vv1 = w_c[c_left]
        vv2 = w_ghost[c_right]

        wx_face[i] = 1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        wy_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

def face_gradient_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_node:'float[:]', cellidf:'int[:,:]', 
                     nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', halofid:'int[:]', centerc:'float[:,:]', 
                     centerh:'float[:,:]', vertexn:'float[:,:]', airDiamond:'float[:]', normalf:'float[:,:]',
                     f_1:'float[:,:]', f_2:'float[:,:]', f_3:'float[:,:]', f_4:'float[:,:]', shift:'float[:,:]', 
                     wx_face:'float[:]', wy_face:'float[:]', wz_face:'float[:]', innerfaces:'int[:]', halofaces:'int[:]',
                     dirichletfaces:'int[:]', neumann:'int[:]', periodicfaces:'int[:]'):
    
    for i in innerfaces:
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
    
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = w_node[i_1]
        V_B = w_node[i_2]
        V_C = w_node[i_3]
        V_D = w_node[i_4]
        
        V_L = w_c[c_left]
        V_R = w_c[c_right]

        wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
    
    for i in periodicfaces:
    
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
    
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = w_node[i_1]
        V_B = w_node[i_2]
        V_C = w_node[i_3]
        V_D = w_node[i_4]
        
        V_L = w_c[c_left]
        V_R = w_c[c_right]

        wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
    
    for i in halofaces:
       
        c_left = cellidf[i][0]
        c_right = halofid[i]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = w_node[i_1]
        V_B = w_node[i_2]
        V_C = w_node[i_3]
        V_D = w_node[i_4]
        
        V_L = w_c[c_left]
        V_R = w_halo[c_right]
        
        wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
            
    for i in dirichletfaces:
       
        c_left = cellidf[i][0]
        c_right = i
    
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = w_node[i_1]
        V_B = w_node[i_2]
        V_C = w_node[i_3]
        V_D = w_node[i_4]
        
        V_L = w_c[c_left]
        V_R = w_ghost[c_right]
        
        wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
        
        
    for i in neumann:
     
        c_left = cellidf[i][0]
        c_right = i
    
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = w_node[i_1]
        V_B = w_node[i_2]
        V_C = w_node[i_3]
        V_D = w_node[i_4]
        
        V_L = w_c[c_left]
        V_R = w_ghost[c_right]
        
        wx_face[i] = (f_1[i][0]*(V_A - V_C) + f_2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        wy_face[i] = (f_1[i][1]*(V_A - V_C) + f_2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        wz_face[i] = (f_1[i][2]*(V_A - V_C) + f_2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]

# TODO centertovertex
@stack_array('center') 
def centertovertex_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                      centerc:'float[:,:]', centerh:'float[:,:]', cellidn:'int[:,:]', periodicn:'int[:,:]',
                      haloidn:'int[:,:]', vertexn:'float[:,:]', namen:'int[:]', centergn:'float[:,:,:]', halocentergn:'float[:,:,:]',
                      R_x:'float[:]', R_y:'float[:]', lambda_x:'float[:]',lambda_y:'float[:]', number:'int[:]', 
                      shift:'float[:,:]', nbproc:'int',  w_n:'float[:]'):
   
    w_n[:] = 0.
    
    from numpy import zeros
    nbnode = len(vertexn)
    center = zeros(3)
    
    for i in range(nbnode):
        for j in range(cellidn[i][-1]):
            cell = cellidn[i][j]
            center[:] = centerc[cell][:]
           
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
            
            w_n[i]  += alpha * w_c[cell]
                
        if centergn[i][0][2] != -1: 
            for j in range(len(centergn[i])):
                cell = int(centergn[i][j][-1])
                if cell != -1:
                    center[:] = centergn[i][j][0:3]
                    
                    xdiff = center[0] - vertexn[i][0]
                    ydiff = center[1] - vertexn[i][1]
                    alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                    
                    w_n[i]  += alpha * w_ghost[cell]
        
        #TODO Must be keeped like that checked ok ;)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] + shift[cell][0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                
                w_n[i]  += alpha * w_c[cell]
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] ==44:
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] + shift[cell][1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                
                w_n[i]  += alpha * w_c[cell]
                    
        if namen[i] == 10:
            for j in range(len(halocentergn[i])):
                cell = int(halocentergn[i][j][-1])
                if cell != -1:
                    center[:] = halocentergn[i][j][0:3]
                  
                    xdiff = center[0] - vertexn[i][0]
                    ydiff = center[1] - vertexn[i][1]
                    
                    alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                    
                    w_n[i]  += alpha * w_haloghost[cell]
            
            # if haloidn[i][-1] > 0 :
            for j in range(haloidn[i][-1]):
                cell = haloidn[i][j]
                center[:] = centerh[cell][0:3]
              
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
             
                w_n[i]  += alpha * w_halo[cell]
        

@stack_array('center')
def centertovertex_3d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]', w_haloghost:'float[:]',
                      centerc:'float[:,:]', centerh:'float[:,:]', cellidn:'int[:,:]', periodicn:'int[:,:]', 
                      haloidn:'int[:,:]', vertexn:'float[:,:]', namen:'int[:]', centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', 
                      R_x:'float[:]', R_y:'float[:]', R_z:'float[:]', lambda_x:'float[:]',lambda_y:'float[:]', 
                      lambda_z:'float[:]', number:'int[:]', shift:'float[:,:]', nbproc:'int', w_n:'float[:]'):
    w_n[:] = 0.
    from numpy import zeros
    
    nbnode = len(vertexn)
    center = zeros(3)
    
    for i in range(nbnode):
        
        for j in range(cellidn[i][-1]):
            cell = cellidn[i][j]
            center[:] = centerc[cell][:]
           
            xdiff = center[0] - vertexn[i][0]
            ydiff = center[1] - vertexn[i][1]
            zdiff = center[2] - vertexn[i][2]
            
            alpha = (1. + lambda_x[i]*xdiff + \
                     lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                              lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
            w_n[i]  += alpha * w_c[cell]
                
        if centergn[i][0][3] != -1: 
            for j in range(len(centergn[i])):
                cell = int(centergn[i][j][-1])
                if cell != -1:
                    center[:] = centergn[i][j][0:3]
                    
                    xdiff = center[0] - vertexn[i][0]
                    ydiff = center[1] - vertexn[i][1]
                    zdiff = center[2] - vertexn[i][2]
                    
                    alpha = (1. + lambda_x[i]*xdiff + \
                              lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                      lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                    w_n[i]  += alpha * w_ghost[cell]
                    
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] + shift[cell][0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                zdiff = center[2] - vertexn[i][2]
                
                alpha = (1. + lambda_x[i]*xdiff + \
                             lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                      lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                
                w_n[i]  += alpha * w_c[cell]
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] == 44:
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] + shift[cell][1] - vertexn[i][1]
                zdiff = center[2] - vertexn[i][2]
                
                alpha = (1. + lambda_x[i]*xdiff + \
                             lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                      lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                w_n[i]  += alpha * w_c[cell]
                
        elif vertexn[i][3] == 55 or vertexn[i][3] == 66:
            for j in range(periodicn[i][-1]):
                cell = periodicn[i][j]
                center[:] = centerc[cell][0:3] 
                
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                zdiff = center[2] + shift[cell][2] - vertexn[i][2]
                
                alpha = (1. + lambda_x[i]*xdiff + \
                             lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                      lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                w_n[i]  += alpha * w_c[cell]
                    
        if namen[i] == 10:
            for j in range(len(halocentergn[i])):
                cell = int(halocentergn[i][j][-1])
                if cell != -1:
                    center[:] = halocentergn[i][j][0:3]
                  
                    xdiff = center[0] - vertexn[i][0]
                    ydiff = center[1] - vertexn[i][1]
                    zdiff = center[2] - vertexn[i][2]
                    
                    alpha = (1. + lambda_x[i]*xdiff + \
                             lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                      lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                                                                      
                    w_n[i]  += alpha * w_haloghost[cell]
            
            # if haloidn[i][-1] > 0 :
            for j in range(haloidn[i][-1]):
                cell = haloidn[i][j]
                center[:] = centerh[cell][0:3]
              
                xdiff = center[0] - vertexn[i][0]
                ydiff = center[1] - vertexn[i][1]
                zdiff = center[2] - vertexn[i][2]
                
                alpha = (1. + lambda_x[i]*xdiff + \
                         lambda_y[i]*ydiff + lambda_z[i]*zdiff)/ (number[i] + lambda_x[i]*R_x[i] + \
                                                                  lambda_y[i]*R_y[i] + lambda_z[i]*R_z[i])
                                                                  
                w_n[i]  += alpha * w_halo[cell]

def barthlimiter_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                    w_x:'float[:]', w_y:'float[:]',  w_z:'float[:]', psi:'float[:]', 
                    cellid:'int[:,:]', faceid:'int[:,:]', namef:'int[:]',
                    halofid:'int[:]', centerc:'float[:,:]', centerf:'float[:,:]'):
    
    from numpy import fabs
    
    nbelement = len(w_c)
    val  = 1.
    psi[:] = val

    for i in range(nbelement):
        w_max = w_c[i]
        w_min = w_c[i]

        for j in range(3):
            face = faceid[i][j]
            if namef[face] == 0 or namef[face] > 10:#
            #11 or namef[face] == 22 or namef[face] == 33 or namef[face] == 44:
                w_max = max([w_max, w_c[cellid[face][0]], w_c[cellid[face][1]]])
                w_min = min([w_min, w_c[cellid[face][0]], w_c[cellid[face][1]]])
            elif namef[face] == 1 or namef[face] == 2 or namef[face] == 3 or namef[face] == 4:
                w_max = max([w_max,  w_c[cellid[face][0]], w_ghost[face]])
                w_min = min([w_min,  w_c[cellid[face][0]], w_ghost[face]])
            else:
                w_max = max([w_max,  w_c[cellid[face][0]], w_halo[halofid[face]]])
                w_min = min([w_min,  w_c[cellid[face][0]], w_halo[halofid[face]]])
        
        for j in range(3):
            face = faceid[i][j]

            r_xyz1 = centerf[face][0] - centerc[i][0] 
            r_xyz2 = centerf[face][1] - centerc[i][1]
            
            delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2 
            
            #TODO choice of epsilon
            if fabs(delta2) < 1e-8:
                psi_ij = 1.
            else:
                if delta2 > 0.:
                    value = (w_max - w_c[i]) / delta2
                    psi_ij = min([val, value])
                if delta2 < 0.:
                    value = (w_min - w_c[i]) / delta2
                    psi_ij = min([val, value])

            psi[i] = min([psi[i], psi_ij])


def barthlimiter_3d(h_c:'float[:]', h_ghost:'float[:]', h_halo:'float[:]',
                    h_x:'float[:]', h_y:'float[:]', h_z:'float[:]',
                    psi:'float[:]', cellid:'int[:,:]', faceid:'int[:,:]', namef:'int[:]',
                    halofid:'int[:]', centerc:'float[:,:]', centerf:'float[:,:]'):
   
    from numpy import fabs
    nbelement = len(h_c)
    psi[:] = 1.

    for i in range(nbelement):
        w_max = h_c[i]
        w_min = h_c[i]

        for j in range(4):
            face = faceid[i][j]
            if namef[face] == 0 or namef[face] > 10:
                w_max = max([w_max, h_c[cellid[face][0]], h_c[cellid[face][1]]])
                w_min = min([w_min, h_c[cellid[face][0]], h_c[cellid[face][1]]])
            elif namef[face] == 10:
                w_max = max([w_max,  h_c[cellid[face][0]], h_halo[halofid[face]]])
                w_min = min([w_min,  h_c[cellid[face][0]], h_halo[halofid[face]]])
            else:
                w_max = max([w_max,  h_c[cellid[face][0]], h_ghost[face]])
                w_min = min([w_min,  h_c[cellid[face][0]], h_ghost[face]])
        
        for j in range(4):
            face = faceid[i][j]

            r_xyz1 = centerf[face][0] - centerc[i][0]
            r_xyz2 = centerf[face][1] - centerc[i][1]
            r_xyz3 = centerf[face][2] - centerc[i][2]
            
            delta2 = h_x[i] * r_xyz1 + h_y[i] * r_xyz2 + h_z[i] * r_xyz3
            
            #TODO choice of epsilon
            if fabs(delta2) < 1e-10:
                psi_ij = 1.
            else:
                if delta2 > 0.:
                    value = (w_max - h_c[i]) / delta2
                    psi_ij = min([1., value])
                if delta2 < 0.:
                    value = (w_min - h_c[i]) / delta2
                    psi_ij = min([1., value])

            psi[i] = min([psi[i], psi_ij])
            
            
            
# @inline
def search_element(a:'int[:]', target_value:'int'):
    find = 0
    for val in a:
        if val == target_value:
            find = 1
            break
    return find

def get_triplet_2d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', halofid:'int[:]',
                   haloext:'int[:,:]', namen:'int[:]', oldnamen:'int[:]', volume:'float[:]', 
                   cellnid:'int[:,:]', centerc:'float[:,:]', centerh:'float[:,:]', halonid:'int[:,:]', periodicnid:'int[:,:]', 
                   centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', airDiamond:'float[:]', 
                   lambda_x:'float[:]', lambda_y:'float[:]', number:'int[:]', R_x:'float[:]', R_y:'float[:]', param1:'float[:]', 
                   param2:'float[:]', param3:'float[:]', param4:'float[:]', shift:'float[:,:]', nbelements:'int', loctoglob:'int[:]',
                   BCdirichlet:'int[:]', a_loc:'float[:]', irn_loc:'int32[:]', jcn_loc:'int32[:]',
                   matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    from numpy import zeros
    
    center = zeros(2)
    parameters = zeros(2)
    cmpt = 0

    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
    
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodeidf[i]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center[:] = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center[:] = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center[:] = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                    
                for j in range(periodicnid[nod][-1]):
                    if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
                        center[0] = centerc[periodicnid[nod][j]][0]  + shift[periodicnid[nod][j]][0]
                        center[1] = centerc[periodicnid[nod][j]][1]  
                    if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
                        center[0] = centerc[periodicnid[nod][j]][0]  
                        center[1] = centerc[periodicnid[nod][j]][1]  + shift[periodicnid[nod][j]][1]
                    
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                    center[:] = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
            cmptparam =+1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

        # right cell------------------------------------------------------
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value =  -1. * param1[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
    
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value =  -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
    
    for i in halofaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
        
        c_rightglob = haloext[halofid[i]][0]
        c_right     = halofid[i]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value =  param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value =  param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodeidf[i]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center[:] = centerc[cellnid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center[:] = centergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(centergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center[:] = halocentergn[nod][j][0:2]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                        index = int(halocentergn[nod][j][2])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1

                for j in range(halonid[nod][-1]):
                    center[:] = centerh[halonid[nod][j]][0:2]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
            cmptparam +=1
            
    for i in dirichletfaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

def compute_2dmatrix_size(nodeidf:'int[:,:]', halofid:'int[:]', cellnid:'int[:,:]',  halonid:'int[:,:]', periodicnid:'int[:,:]', 
                        centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', oldnamen:'int[:]', BCdirichlet:'int[:]', 
                        matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    cmpt = 0
    for i in matrixinnerfaces:
        cmpt = cmpt + 1
        
        for nod in nodeidf[i]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
            # if vertexn[nod][3] not in BCdirichlet:
                for j in range(cellnid[nod][-1]):
                    
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                       
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        cmpt = cmpt + 1
                    
                for j in range(periodicnid[nod][-1]):
                    
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                   
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
        
        cmpt = cmpt + 1
        # right cell------------------------------------------------------
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    # elif namef[i] == 10:
    for i in halofaces:
        cmpt = cmpt + 1
        
        cmpt = cmpt + 1
        cmpt = cmpt + 1
        
        for nod in nodeidf[i]:
            if search_element(BCdirichlet, oldnamen[nod]) == 0:  
                for j in range(cellnid[nod][-1]):
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        cmpt = cmpt + 1

                for j in range(halonid[nod][-1]):
                    cmpt = cmpt + 1
                
    for i in dirichletfaces:
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    return cmpt

def compute_3dmatrix_size(nodeidf:'int[:,:]', halofid:'int[:]',  cellnid:'int[:,:]',  halonid:'int[:,:]', periodicnid:'int[:,:]', 
                          centergn:'float[:,:,:]', halocentergn:'float[:,:,:]',  oldnamen:'int[:]',  BCdirichlet:'int[:]', 
                          matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    from numpy import zeros, int64
    cmpt = 0
    nodes = zeros(4, dtype=int64)
    
    for i in matrixinnerfaces:
       
        nodes[0:3] = nodeidf[i][:]
        nodes[3]   = nodeidf[i][2]
        cmpt = cmpt + 1
            
        for nod in nodes:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
                
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                       
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        cmpt = cmpt + 1
                    
                for j in range(periodicnid[nod][-1]):
                        
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                   
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    cmpt = cmpt + 1
        
        cmpt = cmpt + 1
        # right cell------------------------------------------------------
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    for i in halofaces:
        nodes[0:3] = nodeidf[i][:]
        nodes[3]   = nodeidf[i][2]
        
        cmpt = cmpt + 1
        cmpt = cmpt + 1
        
        for nod in nodes:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        cmpt = cmpt + 1
                        
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        cmpt = cmpt + 1

                for j in range(halonid[nod][-1]):
                    cmpt = cmpt + 1
                    
    for i in dirichletfaces:
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    return cmpt

def get_triplet_3d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', halofid:'int[:]', 
                    haloext:'int[:,:]', namen:'int[:]', oldnamen:'int[:]', volume:'float[:]', centergn:'float[:,:,:]',
                    halocentergn:'float[:,:,:]', periodicnid:'int[:,:]',  cellnid:'int[:,:]', centerc:'float[:,:]', centerh:'float[:,:]',
                    halonid:'int[:,:]', airDiamond:'float[:]', lambda_x:'float[:]', lambda_y:'float[:]', lambda_z:'float[:]', number:'int[:]', 
                    R_x:'float[:]', R_y:'float[:]', R_z:'float[:]',  param1:'float[:]', param2:'float[:]', param3:'float[:]', shift:'float[:,:]', 
                    loctoglob:'int[:]', BCdirichlet:'int[:]', a_loc:'float[:]', irn_loc:'int32[:]', jcn_loc:'int32[:]', matrixinnerfaces:'int[:]', 
                    halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    from numpy import zeros, int64
    
    parameters = zeros(4)
    nodes = zeros(4, dtype=int64)
    
    cmpt = 0
    
    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][:]
        nodes[3]   = nodeidf[i][2]

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1 * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center = centerc[cellnid[nod][j]]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value = alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value = alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
                for j in range(periodicnid[nod][-1]):
                    if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
                        center[0] = centerc[periodicnid[nod][j]][0]  + shift[periodicnid[nod][j]][0]
                        center[1] = centerc[periodicnid[nod][j]][1]  
                        center[2] = centerc[periodicnid[nod][j]][2]
                    if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
                        center[0] = centerc[periodicnid[nod][j]][0]  
                        center[1] = centerc[periodicnid[nod][j]][1]  + shift[periodicnid[nod][j]][1]
                        center[2] = centerc[periodicnid[nod][j]][2]
                    if vertexn[nod][3] == 55 or vertexn[nod][3] == 66:
                        center[0] = centerc[periodicnid[nod][j]][0]  
                        center[1] = centerc[periodicnid[nod][j]][1]  
                        center[2] = centerc[periodicnid[nod][j]][2] + shift[periodicnid[nod][j]][2]
                    
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value =  alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    #right cell-----------------------------------                                                                                              
                    value =  -1. * alpha / volume[c_right] * parameters[cmptparam]
                    irn_loc[cmpt] = c_rightglob
                    jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center = centergn[nod][j][0:3]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        zdiff = center[2] - vertexn[nod][2]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                              lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        
                        index = int(centergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                    
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center = halocentergn[nod][j][0:3]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        zdiff = center[2] - vertexn[nod][2]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                              lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        index = int(halocentergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                        #right cell-----------------------------------                                                                                              
                        value = -1. * alpha / volume[c_right] * parameters[cmptparam]
                        irn_loc[cmpt] = c_rightglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
            cmptparam = cmptparam +1
           
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value = param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1

        # right cell------------------------------------------------------
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_leftglob
        value = param3[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
    
        irn_loc[cmpt] = c_rightglob
        jcn_loc[cmpt] = c_rightglob
        value = -1. * param3[i] / volume[c_right]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
    for i in halofaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][:]
        nodes[3]   = nodeidf[i][2]

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        c_rightglob = haloext[halofid[i]][0]
        c_right     = halofid[i]
        
        cmptparam = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldnamen[nod]) == 0: 
                for j in range(cellnid[nod][-1]):
                    center = centerc[cellnid[nod][j]]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value = alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                
                for j in range(halonid[nod][-1]):
                    center = centerh[halonid[nod][j]]
                    xdiff = center[0] - vertexn[nod][0]
                    ydiff = center[1] - vertexn[nod][1]
                    zdiff = center[2] - vertexn[nod][2]
                    alpha = (1. + lambda_x[nod]*xdiff + \
                              lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                          lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                    value = alpha / volume[c_left] * parameters[cmptparam]
                    irn_loc[cmpt] = c_leftglob
                    jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                    a_loc[cmpt] = value
                    cmpt = cmpt + 1
                    
                for j in range(len(centergn[nod])):
                    if centergn[nod][j][-1] != -1:
                        center = centergn[nod][j][0:3]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        zdiff = center[2] - vertexn[nod][2]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                              lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        
                        index = int(centergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = loctoglob[index]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
                    
                for j in range(len(halocentergn[nod])):
                    if halocentergn[nod][j][-1] != -1:
                        center = halocentergn[nod][j][0:3]
                        xdiff = center[0] - vertexn[nod][0]
                        ydiff = center[1] - vertexn[nod][1]
                        zdiff = center[2] - vertexn[nod][2]
                        alpha = (1. + lambda_x[nod]*xdiff + \
                                  lambda_y[nod]*ydiff + lambda_z[nod]*zdiff)/ (number[nod] + lambda_x[nod]*R_x[nod] + \
                                                                              lambda_y[nod]*R_y[nod] + lambda_z[nod]*R_z[nod])
                        value = alpha / volume[c_left] * parameters[cmptparam]
                        index = int(halocentergn[nod][j][3])
                        irn_loc[cmpt] = c_leftglob
                        jcn_loc[cmpt] = haloext[index][0]
                        a_loc[cmpt] = value
                        cmpt = cmpt + 1
            cmptparam = cmptparam +1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1 * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_rightglob
        value = param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
            
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1 * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = -1. * param3[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
            

def get_rhs_loc_2d(cellfid:'int[:,:]', nodeidf:'int[:,:]', oldname:'int[:]', 
                    volume:'float[:]', centergn:'float[:,:,:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]', rhs_loc:'float[:]', 
                    BCdirichlet:'int[:]', centergf:'float[:,:]', matrixinnerfaces:'int[:]',
                    halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
   
    for i in matrixinnerfaces:
        c_right = cellfid[i][1]
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            rhs_loc[c_right] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
            value_right =  V * param2[i] / volume[c_right]
            rhs_loc[c_right] += value_right
                
    for i in halofaces:
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
    # TODO verify
    for i in dirichletfaces:
        
        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if centergn[i_1][0][2] != -1:     
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] += value_left
           
        if centergn[i_2][0][2] != -1: 
            V = Pbordnode[i_2]
            value_left = -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
        
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs_loc[c_left] += value
            
def get_rhs_glob_2d(cellfid:'int[:,:]', nodeidf:'int[:,:]', oldname:'int[:]', 
                    volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]',  rhs:'float[:]',
                    BCdirichlet:'int[:]', centergf:'float[:,:]', matrixinnerfaces:'int[:]',
                    halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    for i in matrixinnerfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            rhs[c_rightglob] += value_right
            
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
            value_right =  V * param2[i] / volume[c_right]
            rhs[c_rightglob] += value_right
                    
    for i in halofaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
        
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        if centergn[i_1][0][2] != -1:     
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
           
        if centergn[i_2][0][2] != -1: 
            V = Pbordnode[i_2]
            value_left = -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
        
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value

def get_rhs_loc_3d(cellfid:'int[:,:]', nodeidf:'int[:,:]', oldname:'int[:]',
                    volume:'float[:]',  centergn:'float[:,:,:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', 
                    Pbordnode:'float[:]', Pbordface:'float[:]', rhs_loc:'float[:]', BCdirichlet:'int[:]',
                    matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
   
    from numpy import zeros, int64
    
    parameters = zeros(4)
    nodes = zeros(4, dtype=int64)

    for i in matrixinnerfaces:
    
        c_left = cellfid[i][0]
        c_right = cellfid[i][1]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1: 
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
                
                value_right = V * parameters[cmpt] / volume[c_right]
                rhs_loc[c_right] += value_right

            cmpt = cmpt +1

    for i in halofaces:
        
        c_left = cellfid[i][0]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
    
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
            cmpt = cmpt +1
    
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        
        cmpt = 0
        for nod in nodes:
            if centergn[nod][0][3] != -1:    
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
            
            cmpt +=1
            
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs_loc[c_left] += value

def get_rhs_glob_3d(cellfid:'int[:,:]', nodeidf:'int[:,:]', oldname:'int[:]', 
                    volume:'float[:]',  centergn:'float[:,:,:]', loctoglob:'int[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]', rhs:'float[:]', BCdirichlet:'int[:]',
                    matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       

    from numpy import zeros, int64
    parameters = zeros(4)
    nodes = zeros(4, dtype=int64)
    
    for i in matrixinnerfaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1: 
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
                
                value_right = V * parameters[cmpt] / volume[c_right]
                rhs[c_rightglob] += value_right

            cmpt = cmpt +1
    
    for i in halofaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in nodes:
            if search_element(BCdirichlet, oldname[nod]) == 1: 
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
            cmpt = cmpt +1
    
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        nodes[0:3] = nodeidf[i][0:3]
        nodes[3]   = nodeidf[i][2]
        
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in nodes:
            if centergn[nod][0][3] != -1:    
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
                
            cmpt = cmpt +1
            
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value

def compute_P_gradient_2d(P_c:'float[:]', P_ghost:'float[:]', P_halo:'float[:]', P_node:'float[:]', cellidf:'int[:,:]', 
                          nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', halofid:'int[:]', centerc:'float[:,:]', 
                          centerh:'float[:,:]', oldname:'int[:]', airDiamond:'float[:]', f_1:'float[:,:]', f_2:'float[:,:]',
                          f_3:'float[:,:]', f_4:'float[:,:]', normalf:'float[:,:]', shift:'float[:,:]', Pbordnode:'float[:]', Pbordface:'float[:]', 
                          Px_face:'float[:]', Py_face:'float[:]', Pz_face:'float[:]', BCdirichlet:'int[:]', innerfaces:'int[:]',
                          halofaces:'int[:]', neumannfaces:'int[:]', dirichletfaces:'int[:]', periodicfaces:'int[:]'):
    
    for i in innerfaces:
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]

        vv1 = P_c[c_left]
        vv2 = P_c[c_right]
        
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
     
    for i in periodicfaces:
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]

        vv1 = P_c[c_left]
        vv2 = P_c[c_right]
        
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    for i in neumannfaces:
        
        c_left = cellidf[i][0]
        c_right = i
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
            
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]

        vv1 = P_c[c_left]
        vv2 = P_ghost[c_right]
            
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
            
    for i in halofaces:

        c_left = cellidf[i][0]
        c_right = halofid[i]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            vi2 = Pbordnode[i_2]
        
        vv1 = P_c[c_left]
        vv2 = P_halo[c_right]
        
        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])
    
    for i in dirichletfaces:
        
        c_left = cellidf[i][0]
        c_right = i
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = Pbordnode[i_1]
        vi2 = Pbordnode[i_2]
        vv1 = P_c[c_left]
        
        VK = Pbordface[i]
        vv2 = 2. * VK - vv1

        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])


def compute_P_gradient_3d(val_c:'float[:]', v_ghost:'float[:]', v_halo:'float[:]', v_node:'float[:]', cellidf:'int[:,:]', 
                          nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', halofid:'int[:]', centerc:'float[:,:]', 
                          centerh:'float[:,:]', oldname:'int[:]', airDiamond:'float[:]', n1:'float[:,:]', n2:'float[:,:]',
                          n3:'float[:,:]', n4:'float[:,:]', normalf:'float[:,:]', shift:'float[:,:]', Pbordnode:'float[:]', Pbordface:'float[:]', 
                          Px_face:'float[:]', Py_face:'float[:]', Pz_face:'float[:]', BCdirichlet:'int[:]', innerfaces:'int[:]',
                          halofaces:'int[:]', neumannfaces:'int[:]', dirichletfaces:'int[:]', periodicfaces:'int[:]'):
    
    for i in innerfaces:
       
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = v_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if search_element(BCdirichlet, oldname[i_3]) == 1: 
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if search_element(BCdirichlet, oldname[i_4]) == 1: 
            V_D = Pbordnode[i_4]
        
        V_L = val_c[c_left]
        V_R = val_c[c_right]

        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
    
    for i in periodicfaces:
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = v_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if search_element(BCdirichlet, oldname[i_3]) == 1: 
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if search_element(BCdirichlet, oldname[i_4]) == 1: 
            V_D = Pbordnode[i_4]
        
        V_L = val_c[c_left]
        V_R = val_c[c_right]

        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
    
    for i in neumannfaces:
        c_left = cellidf[i][0]
        c_right = i
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = v_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if search_element(BCdirichlet, oldname[i_3]) == 1: 
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if search_element(BCdirichlet, oldname[i_4]) == 1: 
            V_D = Pbordnode[i_4]
        
        V_L = val_c[c_left]
        V_R = v_ghost[c_right]

        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
        
    for i in halofaces:
        c_left = cellidf[i][0]
        c_right = halofid[i]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = v_node[i_1]
        if search_element(BCdirichlet, oldname[i_1]) == 1: 
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if search_element(BCdirichlet, oldname[i_2]) == 1: 
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if search_element(BCdirichlet, oldname[i_3]) == 1: 
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if search_element(BCdirichlet, oldname[i_4]) == 1: 
            V_D = Pbordnode[i_4]
            
        V_L = val_c[c_left]
        V_R = v_halo[c_right]
        
        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
            
    for i in dirichletfaces:   
        c_left = cellidf[i][0]
        c_right = i
    
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = Pbordnode[i_1]
        V_B = Pbordnode[i_2]
        V_C = Pbordnode[i_3]
        V_D = Pbordnode[i_4]
        
        V_L = val_c[c_left]
        V_K = Pbordface[i]
        V_R = 2. * V_K - V_L
        
        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
        
def facetocell(u_face:'float[:]', u_c:'float[:]', faceidc:'int[:,:]', dim:'int'):
  
    nbelements = len(u_c)
    u_c[:] = 0.
    
    for i in range(nbelements):
        for j in range(dim+1):
            u_c[i]  += u_face[faceidc[i][j]]
    
    u_c[:]  /= (dim+1)
