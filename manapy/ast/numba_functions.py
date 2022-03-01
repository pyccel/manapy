from numba import njit
from numpy import zeros

@njit
def search_element2(a, target_value):
    find = 0
    for val in a:
        if val == target_value:
            find = 1
            break
    return find

@njit
def get_triplet_2d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', namef:'int[:]', halofid:'int[:]', 
                   haloext, namen:'int[:]', volume:'float[:]', volumeh, centerg:'float[:,:,:]', 
                   cellnid:'int[:,:]', centerc:'float[:,:]', centerh, halonid:'int[:,:]', periodicnid:'int[:,:]', 
                   centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', airDiamond:'float[:]', 
                   lambda_x:'float[:]', lambda_y:'float[:]', number:'int[:]', R_x:'float[:]', R_y:'float[:]', param1:'float[:]', 
                   param2:'float[:]', param3:'float[:]', param4:'float[:]', shift:'float[:,:]', nbelements:'int', loctoglob:'int[:]',
                   BCdirichlet:'int[:]', a_loc:'float[:]', irn_loc:'int[:]', jcn_loc:'int[:]',
                   matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    center = zeros(2)
    parameters = zeros(2)
    cmpt = 0
    for i in matrixinnerfaces:#innerfaces, periodicinfaces, periodicupperfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
    
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        irn_loc[cmpt] = c_leftglob
        jcn_loc[cmpt] = c_leftglob
        value = param1[i] / volume[c_left]
        a_loc[cmpt] = value
        cmpt = cmpt + 1
        
        cmptparam = 0
        for nod in i_1, i_2:
            if vertexn[nod][3] not in BCdirichlet:
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
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        parameters[0] = param4[i]; parameters[1] = param2[i]
        # elif namef[i] == 10:
        
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
        for nod in i_1, i_2:
            if vertexn[nod][3] not in BCdirichlet:
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
        
    ## TODO verify
    # elif (namef[i] != 6 and namef[i] != 8):# and namef[i] != 3 and namef[i] != 4):
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

@njit
def compute_2dmatrix_size(nodeidf:'int[:,:]', vertexn:'float[:,:]', namef:'int[:]', halofid:'int[:]', 
                        cellnid:'int[:,:]',  halonid:'int[:,:]', periodicnid:'int[:,:]', 
                        centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', 
                        BCdirichlet:'int[:]', matrixinnerfaces:'int[:]',  
                        halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    cmpt = 0
    for i in matrixinnerfaces:#innerfaces, periodicinfaces, periodicupperfaces:
        cmpt = cmpt + 1
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        for nod in i_1, i_2:
            if vertexn[nod][3] not in BCdirichlet:
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
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]  
        
        cmpt = cmpt + 1
        cmpt = cmpt + 1
        
        for nod in i_1, i_2:
            if vertexn[nod][3] not in BCdirichlet:
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
                
    # TODO verify
    for i in dirichletfaces:
    # elif (namef[i] != 6 and namef[i] != 8):# and namef[i] != 3 and namef[i] != 4):
            
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    return cmpt


@njit
def compute_3dmatrix_size(nodeidf:'int[:,:]', halofid:'int[:]', 
                          cellnid:'int[:,:]',  halonid:'int[:,:]', periodicnid:'int[:,:]', 
                          centergn:'float[:,:,:]', halocentergn:'float[:,:,:]', oldname:'int[:]',
                          BCdirichlet:'int[:]', matrixinnerfaces:'int[:]',
                          halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    cmpt = 0
    for i in matrixinnerfaces:#innerfaces, periodicinfaces, periodicupperfaces:
       
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        i_3 = nodeidf[i][2]
        i_4 = i_3
        cmpt = cmpt + 1
            
        for nod in i_1, i_2, i_3, i_4:
            if oldname[nod] not in BCdirichlet:
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
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        i_3 = nodeidf[i][2]
        i_4 = i_3
        
        cmpt = cmpt + 1
        cmpt = cmpt + 1
        
        for nod in i_1, i_2, i_3, i_4:
            if oldname[nod] not in BCdirichlet:
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
                    
        # TODO verify
    for i in dirichletfaces:
        cmpt = cmpt + 1
        cmpt = cmpt + 1
            
    return cmpt

@njit
def get_triplet_3d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', halofid:'int[:]', 
                   haloext, namen:'int[:]', oldname:'int[:]', volume:'float[:]', centergn:'float[:,:,:]',
                   halocentergn:'float[:,:,:]', periodicnid:'int[:,:]', 
                   cellnid:'int[:,:]', centerc:'float[:,:]', centerh:'float[:,:]', halonid:'int[:,:]', airDiamond:'float[:]', 
                   lambda_x:'float[:]', lambda_y:'float[:]', lambda_z:'float[:]', number:'int[:]', R_x:'float[:]', R_y:'float[:]', 
                   R_z:'float[:]',  param1:'float[:]', param2:'float[:]', param3:'float[:]', shift:'float[:,:]', loctoglob:'int[:]',
                   BCdirichlet:'int[:]', a_loc:'float[:]', irn_loc:'int[:]', jcn_loc:'int[:]',matrixinnerfaces:'int[:]', 
                   halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    parameters = zeros(4)
    cmpt = 0
    
    for i in matrixinnerfaces:#innerfaces, periodicinfaces, periodicupperfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3

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
        for nod in i_1, i_2, i_3, i_4:
            if vertexn[nod][3] not in BCdirichlet:
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
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3

        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        c_rightglob = haloext[halofid[i]][0]
        c_right     = halofid[i]
        
        cmptparam = 0
        for nod in i_1, i_2, i_3, i_4:
            if vertexn[nod][3] not in BCdirichlet:
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
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3

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
            

@njit
def get_rhs_loc_2d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', namef:'int[:]',
                   volume:'float[:]', centergn:'float[:,:,:]', param1:'float[:]', param2:'float[:]', 
                   param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]', rhs_loc:'float[:]', 
                   BCdirichlet:'int[:]', centergf:'float[:,:]', matrixinnerfaces:'int[:]',
                   halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
   
    for i in matrixinnerfaces:#innerfaces, periodicinfaces, periodicupperfaces:
        c_right = cellfid[i][1]
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        # reproduire dans faces.name = 10 (centergn[i_1][0][2] astuce)
        if vertexn[i_1][3] in BCdirichlet:
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            rhs_loc[c_right] += value_right
            
        if vertexn[i_2][3] in BCdirichlet:
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
            value_right =  V * param2[i] / volume[c_right]
            rhs_loc[c_right] += value_right
                
    for i in halofaces:
        c_left = cellfid[i][0]
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        
        if vertexn[i_1][3] in BCdirichlet:
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            rhs_loc[c_left] += value_left
            
        if vertexn[i_2][3] in BCdirichlet:
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
        
        # vi1 = Pbordnode[i_1]
        # vi2 = Pbordnode[i_2]
        # gamma = centergf[i][2]
        V_K = Pbordface[i]
        #gamma * vi1 + (1.-gamma) * vi2;
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs_loc[c_left] += value
            
@njit
def get_rhs_glob_2d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', namef:'int[:]',
                    volume:'float[:]', centergn:'float[:,:,:]', loctoglob:'int[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', param4:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]',  rhs:'float[:]',
                    BCdirichlet:'int[:]', centergf:'float[:,:]', matrixinnerfaces:'int[:]',
                    halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
    
    for i in matrixinnerfaces:#periodicinfaces, periodicupperfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]    
        
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        if vertexn[i_1][3] in BCdirichlet:
            V = Pbordnode[i_1]
            value_left = -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] +=  value_left
            
            value_right = V * param4[i] / volume[c_right]
            rhs[c_rightglob] += value_right
            
        if vertexn[i_2][3] in BCdirichlet:
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
        
        if vertexn[i_1][3] in BCdirichlet:
            V = Pbordnode[i_1]
            value_left =  -1. * V * param4[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
        if vertexn[i_2][3] in BCdirichlet:
            V = Pbordnode[i_2]
            value_left =  -1. * V * param2[i] / volume[c_left]
            rhs[c_leftglob] += value_left
            
    # TODO verify (checked ok ;))
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
        
        # vi1 = Pbordnode[i_1]
        # vi2 = Pbordnode[i_2]
        # gamma = centergf[i][2]
        V_K = Pbordface[i]
        #gamma * vi1 + (1.-gamma) * vi2;
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value

@njit
def get_rhs_loc_3d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', namef:'int[:]', 
                    volume:'float[:]',  centergn:'float[:,:,:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', 
                    Pbordnode:'float[:]', Pbordface:'float[:]', rhs_loc:'float[:]', BCdirichlet:'int[:]',
                    matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       
   
    parameters = zeros(4)
    
    for i in matrixinnerfaces:#periodicinfaces, periodicupperfaces:
    
        c_left = cellfid[i][0]
        c_right = cellfid[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in i_1, i_2, i_3, i_4:
            if vertexn[nod][3] in BCdirichlet:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
                
                value_right = V * parameters[cmpt] / volume[c_right]
                rhs_loc[c_right] += value_right

            cmpt = cmpt +1

    for i in halofaces:
        
        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
    
        cmpt = 0
        for nod in i_1, i_2, i_3, i_4:
            if vertexn[nod][3] in BCdirichlet:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
            cmpt = cmpt +1
    
    
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        
        cmpt = 0
        for nod in i_1, i_2, i_3, i_4:
            if centergn[nod][0][3] != -1:    
            # if vertexn[nod][3] != 0:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs_loc[c_left] += value_left
            
            cmpt +=1
            
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs_loc[c_left] += value

@njit
def get_rhs_glob_3d(cellfid:'int[:,:]', nodeidf:'int[:,:]', vertexn:'float[:,:]', namef:'int[:]',
                    volume:'float[:]',  centergn:'float[:,:,:]', loctoglob:'int[:]', param1:'float[:]', param2:'float[:]', 
                    param3:'float[:]', Pbordnode:'float[:]', Pbordface:'float[:]', rhs:'float[:]', BCdirichlet:'int[:]',
                    matrixinnerfaces:'int[:]', halofaces:'int[:]', dirichletfaces:'int[:]'):                                                                                                                                                                       

    parameters = zeros(4)
    for i in matrixinnerfaces:#periodicinfaces, periodicupperfaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
            
        c_right = cellfid[i][1]
        c_rightglob = loctoglob[c_right]
        
        cmpt = 0
        for nod in i_1, i_2, i_3, i_4:
            if vertexn[nod][3] in BCdirichlet:#if vertexn[nod][3] != 0:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
                
                value_right = V * parameters[cmpt] / volume[c_right]
                rhs[c_rightglob] += value_right

            cmpt = cmpt +1
    
    for i in halofaces:
        
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in i_1, i_2, i_3, i_4:
            if vertexn[nod][3] in BCdirichlet:
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
            cmpt = cmpt +1
    
    for i in dirichletfaces:
        c_left = cellfid[i][0]
        c_leftglob  = loctoglob[c_left]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1] 
        i_3 = nodeidf[i][2]
        i_4 = i_3
        parameters[0] = param1[i]; parameters[1] = param2[i]
        parameters[2] = -1. * param1[i]; parameters[3] = -1. * param2[i]
        
        cmpt = 0
        for nod in i_1, i_2, i_3, i_4:
            # if vertexn[nod][3] != 0:
            if centergn[nod][0][3] != -1:    
                V = Pbordnode[nod]
                value_left = -1. * V * parameters[cmpt] / volume[c_left]
                rhs[c_leftglob] += value_left
                
            cmpt = cmpt +1
            
        V_K = Pbordface[i]
        value = -2. * param3[i] / volume[c_left] * V_K
        rhs[c_leftglob] += value

@njit
def compute_P_gradient_2d(P_c:'float[:]', P_ghost:'float[:]', P_halo:'float[:]', P_node:'float[:]', cellidf:'int[:,:]', 
                          nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', halofid:'int[:]', centerc:'float[:,:]', 
                          centerh:'float[:,:]', vertexn:'float[:,:]', airDiamond:'float[:]', f_1:'float[:,:]', f_2:'float[:,:]',
                          f_3:'float[:,:]', f_4:'float[:,:]', normalf:'float[:,:]', shift:'float[:,:]', Pbordnode:'float[:]', Pbordface:'float[:]', 
                          Px_face:'float[:]', Py_face:'float[:]', Pz_face:'float[:]', BCdirichlet:'int[:]', innerfaces:'int[:]',
                          halofaces:'int[:]', neumannfaces:'int[:]', dirichletfaces:'int[:]', periodicfaces:'int[:]'):
    
    for i in innerfaces:
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]
        
        vi1 = P_node[i_1]
        if vertexn[i_1][3] in BCdirichlet: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if vertexn[i_2][3] in BCdirichlet: 
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
        if vertexn[i_1][3] in BCdirichlet: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if vertexn[i_2][3] in BCdirichlet: 
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
        if vertexn[i_1][3] in BCdirichlet: 
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if vertexn[i_2][3] in BCdirichlet: 
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
        if vertexn[i_1][3] in BCdirichlet:
            vi1 = Pbordnode[i_1]
        vi2 = P_node[i_2]
        if vertexn[i_2][3] in BCdirichlet:
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
        # gamma = centergf[i][2]
        VK = Pbordface[i]#gamma * vi1 + (1.-gamma) * vi2;
        vv2 = 2. * VK - vv1

        Px_face[i] = -1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][1] + (vv1 + vi2)*f_2[i][1] + (vi2 + vv2)*f_3[i][1] + (vv2 + vi1)*f_4[i][1])
        Py_face[i] =  1/(2*airDiamond[i])*((vi1 + vv1)*f_1[i][0] + (vv1 + vi2)*f_2[i][0] + (vi2 + vv2)*f_3[i][0] + (vv2 + vi1)*f_4[i][0])

@njit
def compute_P_gradient_3d(v_c:'float[:]', v_ghost:'float[:]', v_halo:'float[:]', v_node:'float[:]', cellidf:'int[:,:]', 
                          nodeidf:'int[:,:]', centergf:'float[:,:]', namef:'int[:]', halofid:'int[:]', centerc:'float[:,:]', 
                          centerh:'float[:,:]', vertexn:'float[:,:]', airDiamond:'float[:]', n1:'float[:,:]', n2:'float[:,:]',
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
        if vertexn[i_1][3] in BCdirichlet:
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if vertexn[i_2][3] in BCdirichlet:
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if vertexn[i_3][3] in BCdirichlet:
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if vertexn[i_4][3] in BCdirichlet:
            V_D = Pbordnode[i_4]
        
        V_L = v_c[c_left]
        V_R = v_c[c_right]

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
        if vertexn[i_1][3] in BCdirichlet:
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if vertexn[i_2][3] in BCdirichlet:
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if vertexn[i_3][3] in BCdirichlet:
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if vertexn[i_4][3] in BCdirichlet:
            V_D = Pbordnode[i_4]
        
        V_L = v_c[c_left]
        V_R = v_c[c_right]

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
        if vertexn[i_1][3] in BCdirichlet:
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if vertexn[i_2][3] in BCdirichlet:
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if vertexn[i_3][3] in BCdirichlet:
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if vertexn[i_4][3] in BCdirichlet:
            V_D = Pbordnode[i_4]
        
        V_L = v_c[c_left]
        V_R = v_ghost[c_right]

        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]
        
    for i in halofaces:
        # print("halo", i, namef[i])
        c_left = cellidf[i][0]
        c_right = halofid[i]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        
        V_A = v_node[i_1]
        if vertexn[i_1][3] in BCdirichlet:
            V_A = Pbordnode[i_1]
        V_B = v_node[i_2]
        if vertexn[i_2][3] in BCdirichlet:
            V_B = Pbordnode[i_2]
        V_C = v_node[i_3]
        if vertexn[i_3][3] in BCdirichlet:
            V_C = Pbordnode[i_3]
        V_D = v_node[i_4]
        if vertexn[i_4][3] in BCdirichlet:
            V_D = Pbordnode[i_4]   
            
        V_L = v_c[c_left]
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
        
        V_L = v_c[c_left]
        V_K = Pbordface[i]
        V_R = 2. * V_K - V_L
        
        Px_face[i] = -1. * (n1[i][0]*(V_A - V_C) + n2[i][0]*(V_B - V_D) + normalf[i][0]*(V_R - V_L)) / airDiamond[i]
        Py_face[i] = -1. * (n1[i][1]*(V_A - V_C) + n2[i][1]*(V_B - V_D) + normalf[i][1]*(V_R - V_L)) / airDiamond[i]
        Pz_face[i] = -1. * (n1[i][2]*(V_A - V_C) + n2[i][2]*(V_B - V_D) + normalf[i][2]*(V_R - V_L)) / airDiamond[i]

@njit
def convert_solution(x1:'float[:]', x1converted:'float[:]', tc:'int[:]', b0Size:'int'):
    for i in range(b0Size):
        x1converted[i] = x1[tc[i]]

@njit
def rhs_value_dirichlet(Pbordnode:'float[:]', Pbordface:'float[:]', nodes:'int[:]', faces:'int[:]', value:'float'):
    
    for i in faces:
        Pbordface[i] = value
    for i in nodes:
        Pbordnode[i] = value

@njit
def ghost_value_neumann(w_c:'float[:]', w_ghost:'float[:]', cellid:'int[:,:]', faces:'int[:]'):
    
    for i in faces:
        w_ghost[i]  = w_c[cellid[i][0]]
     
@njit
def ghost_value_dirichlet(value:'float', w_ghost:'float[:]', cellid:'int[:,:]', faces:'int[:]'):
    
    for i in faces:
        w_ghost[i]  = value


@njit
def haloghost_value_neumann(w_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                            BCindex: 'int', halonodes:'int[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellhalo  = int(haloghostcenter[i][j][-3])
                    cellghost = int(haloghostcenter[i][j][-1])
    
                    w_haloghost[cellghost]   = w_halo[cellhalo]
                
                
@njit
def haloghost_value_dirichlet(value:'float', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                              BCindex: 'int', halonodes:'int[:]'):

    for i in halonodes:
        for j in range(len(haloghostcenter[i])):

            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellghost = int(haloghostcenter[i][j][-1])
                    w_haloghost[cellghost]   = value
