#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:04:50 2020

@author: kissami
"""
import meshio
from mpi4py import MPI
import numpy as np
from numba import njit, jit
from manapy import ddm


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

@njit(fastmath=True)
def add(sol_1, sol_2):
    sol_1.h += sol_2.h
    sol_1.hu += sol_2.hu
    sol_1.hv += sol_2.hv
    sol_1.hc += sol_2.hc
    sol_1.Z += sol_2.Z

    return sol_1

@njit(fastmath=True)
def minus(sol_1, sol_2):
    sol_1.h -= sol_2.h
    sol_1.hu -= sol_2.hu
    sol_1.hv -= sol_2.hv
    sol_1.hc -= sol_2.hc
    sol_1.Z -= sol_2.Z

    return sol_1

@jit(fastmath=True)#nopython = False)
def matmul(rmatrix, matrix1, matrix2):

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix

@njit(fastmath=True)
def centertovertex(w_c, w_ghost, w_halo, centerc, centerh, cellidn, haloidn, vertexn, namen,
                   w_node, uexact, unode_exacte):
    I_xx = np.zeros(len(vertexn))
    I_yy = np.zeros(len(vertexn))
    I_xy = np.zeros(len(vertexn))
    R_x = np.zeros(len(vertexn))
    R_y = np.zeros(len(vertexn))
    lambda_x = np.zeros(len(vertexn))
    lambda_y = np.zeros(len(vertexn))
    number = np.zeros(len(vertexn))
    
    for i in range(len(vertexn)):
        for j in range(len(cellidn[i])):
            if cellidn[i][j] != -1:
                center = centerc[cellidn[i][j]]
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] = number[i] + 1
        
        if SIZE > 1:
            for j in range(len(haloidn[i])):
                cell = haloidn[i][j]
                if cell != -1:
                    center = centerh[cell]
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

    for i in range(len(vertexn)):
        for j in range(len(cellidn[i])):
            if cellidn[i][j] != -1:     
                if namen[i] == 0:
                    center = centerc[cellidn[i][j]]
                    xdiff = center[0] - vertexn[i][0]
                    ydiff = center[1] - vertexn[i][1]
                    alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                    w_node[i].h +=  alpha * w_c[cellidn[i][j]].h
                    w_node[i].hu += alpha * w_c[cellidn[i][j]].hu
                    w_node[i].hv += alpha * w_c[cellidn[i][j]].hv
                    w_node[i].hc += alpha * w_c[cellidn[i][j]].hc
                    w_node[i].Z += alpha * w_c[cellidn[i][j]].Z
                    unode_exacte[i] += alpha * uexact[cellidn[i][j]]
                else:
                    alpha = 1./number[i]
                    w_node[i].h +=  alpha * w_c[cellidn[i][j]].h
                    w_node[i].hu += alpha * w_c[cellidn[i][j]].hu
                    w_node[i].hv += alpha * w_c[cellidn[i][j]].hv
                    w_node[i].hc += alpha * w_c[cellidn[i][j]].hc
                    w_node[i].Z += alpha * w_c[cellidn[i][j]].Z
                    unode_exacte[i] += alpha * uexact[cellidn[i][j]]

        if SIZE > 1:
            for j in range(len(haloidn[i])):
                cell = haloidn[i][j]
                if cell != -1:
                    center = centerh[cell]
                    xdiff = center[0] - vertexn[i][0]
                    ydiff = center[1] - vertexn[i][1]
                    alpha = (1. + lambda_x[i]*xdiff + lambda_y[i]*ydiff)/(number[i] + lambda_x[i]*R_x[i] + lambda_y[i]*R_y[i])
                    w_node[i].h +=  alpha * w_halo[cell].h
                    w_node[i].hu += alpha * w_halo[cell].hu
                    w_node[i].hv += alpha * w_halo[cell].hv
                    w_node[i].hc += alpha * w_halo[cell].hc
                    w_node[i].Z += alpha * w_halo[cell].Z

    return w_node, unode_exacte

@njit(fastmath=True)
def derivxy(w_c, w_ghost, w_halo, centerc, centerh, cellnid, halonid, nodeid, w_x, w_y):


    for i in range(len(centerc)):
        i_xx = 0.
        i_yy = 0.
        i_xy = 0.
        j_xh = 0.
        j_yh = 0

        j_xhu = 0.
        j_yhu = 0.
        j_xhv = 0.
        j_yhv = 0.
        j_xhc = 0.
        j_yhc = 0.
        j_xz = 0.
        j_yz = 0.


        for k in range(3):
            nod = nodeid[i][k]
            for j in range(len(cellnid[nod])):
                cell = cellnid[nod][j]
                if cell != -1:
                    j_x = centerc[cell][0] - centerc[i][0]
                    j_y = centerc[cell][1] - centerc[i][1]
                    i_xx += pow(j_x, 2)
                    i_yy += pow(j_y, 2)
                    i_xy += (j_x * j_y)

                    j_xh += (j_x * (w_c[cell].h - w_c[i].h))
                    j_yh += (j_y * (w_c[cell].h - w_c[i].h))

                    j_xhu += (j_x * (w_c[cell].hu - w_c[i].hu))
                    j_yhu += (j_y * (w_c[cell].hu - w_c[i].hu))

                    j_xhv += (j_x * (w_c[cell].hv - w_c[i].hv))
                    j_yhv += (j_y * (w_c[cell].hv - w_c[i].hv))

                    j_xhc += (j_x * (w_c[cell].hc - w_c[i].hc))
                    j_yhc += (j_y * (w_c[cell].hc - w_c[i].hc))

                    j_xz += (j_x * (w_c[cell].Z - w_c[i].Z))
                    j_yz += (j_y * (w_c[cell].Z - w_c[i].Z))


            if SIZE > 1:
                for j in range(len(halonid[nod])):
                    cell = halonid[nod][j]
                    if cell != -1:
                        j_x = centerh[cell][0] - centerc[i][0]
                        j_y = centerh[cell][1] - centerc[i][1]
                        i_xx += pow(j_x, 2)
                        i_yy += pow(j_y, 2)
                        i_xy += (j_x * j_y)

                        j_xh += (j_x * (w_halo[cell].h - w_c[i].h))
                        j_yh += (j_y * (w_halo[cell].h - w_c[i].h))

                        j_xhu += (j_x * (w_halo[cell].hu - w_c[i].hu))
                        j_yhu += (j_y * (w_halo[cell].hu - w_c[i].hu))

                        j_xhv += (j_x * (w_halo[cell].hv - w_c[i].hv))
                        j_yhv += (j_y * (w_halo[cell].hv - w_c[i].hv))

                        j_xhc += (j_x * (w_halo[cell].hc - w_c[i].hc))
                        j_yhc += (j_y * (w_halo[cell].hc - w_c[i].hc))

                        j_xz += (j_x * (w_halo[cell].Z - w_c[i].Z))
                        j_yz += (j_y * (w_halo[cell].Z - w_c[i].Z))


        dia = i_xx * i_yy - pow(i_xy, 2)

        w_x[i].h = (i_yy * j_xh - i_xy * j_yh) / dia
        w_y[i].h = (i_xx * j_yh - i_xy * j_xh) / dia

        w_x[i].hu = (i_yy * j_xhu - i_xy * j_yhu) / dia
        w_y[i].hu = (i_xx * j_yhu - i_xy * j_xhu) / dia

        w_x[i].hv = (i_yy * j_xhv - i_xy * j_yhv) / dia
        w_y[i].hv = (i_xx * j_yhv - i_xy * j_xhv) / dia

        w_x[i].hc = (i_yy * j_xhc - i_xy * j_yhc) / dia
        w_y[i].hc = (i_xx * j_yhc - i_xy * j_xhc) / dia

        w_x[i].Z = (i_yy * j_xz - i_xy * j_yz) / dia
        w_y[i].Z = (i_xx * j_yz - i_xy * j_xz) / dia

    return w_x, w_y
@njit(fastmath=True)
def update_fluxes(w_c, w_ghost, w_halo, w_x, w_y, wx_halo, wy_halo, nodeidc, faceidc,
                   cellidc, centerc, volumec, cellidf, nodeidf, normalf, centerf, namef, halofid,
                   vertexn, centerh, mystruct, order, term_convec, term_source):
    
    if (term_convec == "on"):
        #update the rezidus using explicit scheme
        rezidus = ddm.explicitscheme(w_c, w_x, w_y, w_ghost, w_halo, wx_halo, wy_halo, cellidf, 
                                 faceidc, centerc, centerh, centerf, normalf, halofid,
                                 namef, mystruct, order)
    if (term_source == "on"):
        #update the source using explicit scheme
#        source = ddm.term_source_srnh(w_c, w_ghost, w_halo, w_x, w_y, wx_halo, wy_halo, nodeidc, 
#                                  faceidc, cellidc, centerc, volumec, cellidf, nodeidf, normalf, 
#                                  centerf, namef, vertexn, halofid, centerh, mystruct, order)
        source = ddm.term_source_vasquez(w_c, w_ghost, nodeidc, faceidc, cellidc, centerc, volumec, 
                                     cellidf,  nodeidf, normalf, centerf, vertexn, mystruct)
    return rezidus, source

@njit(fastmath=True)
def update(w_c, wnew, dtime, rez, src, vol):
    for i in range(len(w_c)):

        wnew.h[i] = w_c.h[i]  + dtime  * ((rez["h"][i]  + src["h"][i])/vol[i])
        wnew.hu[i] = w_c.hu[i] + dtime * ((rez["hu"][i] + src["hu"][i])/vol[i])
        wnew.hv[i] = w_c.hv[i] + dtime * ((rez["hv"][i] + src["hv"][i])/vol[i])
        wnew.hc[i] = w_c.hc[i] + dtime * ((rez["hc"][i] + src["hc"][i])/vol[i])
        wnew.Z[i] = w_c.Z[i]  + dtime  * ((rez["Z"][i] + src["Z"][i])/vol[i])

    return wnew

@njit(fastmath=True)
def time_step(w_c, cfl, normal, volume, faceid):

    dt_c = np.zeros(len(faceid))

    for i in range(len(faceid)):
        velson = np.sqrt(9.81*w_c[i].h)
        lam = 0
        for j in range(3):
            norm = normal[faceid[i][j]]
            u_n = np.fabs(w_c.hu[i]/w_c.h[i]*norm[0] + w_c.hv[i]/w_c.h[i]*norm[1])
            mes = np.sqrt(norm[0]*norm[0] + norm[1]*norm[1])
            lam_j = u_n/mes + velson
            lam += lam_j * mes

        dt_c[i] = cfl * volume[i]/lam

    dtime = np.asarray(np.min(dt_c))

    return dtime

@njit(fastmath=True)
def exact_solution(uexact, hexact, zexact, center, time):

#    grav = 9.81
#    h_m = 2.534
#    u_m = 4.03
#    h_1 = 5.
#    h_2 = 1.
#    
#    x_0 = 6.
#    c_o = np.sqrt(grav*h_1)
#    s_s = np.sqrt(0.5*grav*h_m*(1.+(h_m/h_2)))
#
#    x_1 = x_0 - c_o*time
#    x_2 = x_0 + 2.*c_o*time - 3.*np.sqrt(grav*h_m)*time
#    x_3 = s_s*time + x_0


    for i in range(len(center)):
        xcent = center[i][0]
#        if xcent < x_1:
#            hexact[i] = h_1
#            uexact[i] = 0
#
#        elif xcent < x_2 and xcent >= x_1:
#            hexact[i] = 1/(9*grav) * (2*np.sqrt(grav*h_1) - (xcent-x_0)/time)**2
#            uexact[i] = 2/3 * (np.sqrt(grav*h_1) + (xcent-x_0)/time)
#
#        elif xcent < x_3 and xcent >= x_2:
#            hexact[i] = h_m
#            uexact[i] = u_m
#
#        elif xcent >= x_3:
#            hexact[i] = h_2
#            uexact[i] = 0


        if np.fabs(xcent - 1500/2) <= 1500/8 :
            zexact[i] = 8#10 + (40*xcent/14000) + 10*np.sin(np.pi*(4*xcent/14000 - 0.5))
    
        hexact[i] = 20 - zexact[i] - 4*np.sin(np.pi*(4*time/86400 + 0.5))
        #hexact[i] = 64.5 - zexact[i] - 4*np.sin(np.pi*(4*time/86400 + 0.5))
    
        uexact[i] = (xcent - 1500)*np.pi/(5400*hexact[i]) * np.cos(np.pi*(4*time/86400 + 0.5))

    return uexact, hexact

@njit(fastmath=True)
def initialisation(w_c, center):

    nbelements = len(center)
#    sigma1 = 264
#    sigma2 = 264
#    c_1 = 10
#    c_2 = 6.5
#    x_1 = 1400
#    y_1 = 1400
#    x_2 = 2400
#    y_2 = 2400

    choix = 3 # (0,creneau 1:gaussienne)
    if choix == 0:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            if xcent <= 6:
                w_c.h[i] = 5.
            else:
                w_c.h[i] = 1.

            w_c.hu[i] = 10.
            w_c.hv[i] = 0.
            w_c.hc[i] = 0.
            w_c.Z[i] = 0.

    elif choix == 1:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            w_c.h[i] = 1#+np.exp(-30*(pow(xcent-2, 2) + pow(ycent-1.5, 2)))
            w_c.hc[i] = np.exp(-30*(pow(xcent-2, 2) + pow(ycent-1.5, 2)))
#            w_c.hc[i] = c_1 * np.exp(-1* (pow(xcent - x_1, 2)
#                + pow(ycent -y_1, 2))/ pow(sigma1, 2)) + c_2 * np.exp(-1* (pow(xcent - x_2, 2)
#                + pow(ycent -y_2, 2))/ pow(sigma2, 2))
            w_c.hu[i] = 5.
            w_c.hv[i] = 0.
            w_c.Z[i] = 0.

    elif choix == 2:
 #       L = 14000
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            w_c.Z[i] = 0.8 * np.exp(-5*(xcent - 1)**2 - 50* (ycent - 0.5)**2) #
            #w_c.Z[i] = 10 + (40*xcent/L) + 10*np.sin(np.pi*(4*xcent/L - 0.5))
            w_c.h[i] = 1 - w_c.Z[i]
            #w_c.h[i] = 60.5 - w_c.Z[i]

            w_c.hu[i] = 0.
            w_c.hv[i] = 0.
            w_c.hc[i] = 0.

    elif choix == 3:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            w_c.Z[i] = 0

            if np.fabs(xcent - 1500/2) <= 1500/8:
                w_c.Z[i] = 8

            w_c.h[i] = 16 - w_c.Z[i]
            w_c.hu[i] = 0.
            w_c.hv[i] = 0.
            w_c.hc[i] = 0.


    elif choix == 4:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            w_c.Z[i] = (10*np.exp(-1*((xcent-2)**2+(ycent-6)**2)) + 15*np.exp(-1*((xcent-2.5)**2+(ycent-2.5)**2)))/20
            w_c.Z[i] += (12*np.exp(-1*((xcent-5)**2+(ycent-5)**2)) + 6*np.exp(-2*((xcent-7.5)**2+(ycent-7.5)**2)))/20
            w_c.Z[i] += (16*np.exp(-1*((xcent-7.5)**2+(ycent-2)**2)))/20

            w_c.h[i] = 1 - w_c.Z[i]

            w_c.hu[i] = 0.
            w_c.hv[i] = 0.
            w_c.hc[i] = 0.

    return w_c

@njit
def ghost_value(w_c, w_ghost, cellid, name, normal, time):

    L = 86400
    for i in range(len(cellid)):
        w_ghost[i] = w_c[cellid[i][0]]
        aa = 4*time/L + 0.5

        if name[i] == 1:
            w_ghost[i].h = 20 - 4*np.sin(np.pi * aa)
            u = -1500 * np.pi/(5400*w_ghost[i].h) * np.cos(np.pi * aa)
            v = 0

            w_ghost[i].hu = w_ghost[i].h * u
            w_ghost[i].hv = w_ghost[i].h * v

        elif name[i] == 2:
            w_ghost[i].hu = 0
            w_ghost[i].hv = 0

        if (name[i] == 3 or name[i] == 4):
            #slip conditions
            norm = normal[i]
    
            u_i = w_c[cellid[i][0]].hu/w_c[cellid[i][0]].h
            v_i = w_c[cellid[i][0]].hv/w_c[cellid[i][0]].h
    
            mesn = np.sqrt(norm[0]*norm[0]+ norm[1]*norm[1])
    
            s_n = norm / mesn
    
            u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
            v_g = v_i*(s_n[0]*s_n[0] - s_n[1]*s_n[1]) - 2.0*u_i*s_n[0]*s_n[1]
    
            w_ghost[i].h = w_c[cellid[i][0]].h
            w_ghost[i].Z = w_c[cellid[i][0]].Z
            w_ghost[i].hc = w_c[cellid[i][0]].hc
    
            w_ghost[i].hu = w_c[cellid[i][0]].h * u_g
            w_ghost[i].hv = w_c[cellid[i][0]].h * v_g

#        elif (name[i] == 1 or name[i] == 2):
#            w_ghost[i] = w_c[cellid[i][0]]
##        if name[i] !=0:
#            w_ghost[i] = w_c[cellid[i][0]]

    return w_ghost

def save_paraview_results(w_c, solexact, niter, miter, time, dtime, rank, size, cells, nodes):

    elements = {"triangle": cells}
    points = []
    for i in nodes:
        points.append([i[0], i[1], i[2]])

    data = {"h" : w_c["h"], "u" : w_c["hu"]/w_c["h"], "v": w_c["hv"]/w_c["h"], "c": w_c["hc"]/w_c["h"],
            "Z": w_c["Z"], "h+Z": w_c["h"] + w_c["Z"], "exact": solexact}
    if len(w_c) == len(cells):
        data = {"h": data, "u":data, "v": data, "c": data, "Z":data, "h+Z":data, "exact":data}

    maxh = np.zeros(1)
    maxh = max(w_c["h"])
    integral_sum = np.zeros(1)

    COMM.Reduce(maxh, integral_sum, MPI.MAX, 0)
    if rank == 0:
        print(" **************************** Computing ****************************")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Iteration = ", niter, "time = ", np.float16(time), "time step = ", np.float16(dtime))
        print("max h =", np.float16(integral_sum[0]))
    
    if len(w_c) == len(cells):
        meshio.write_points_cells("results/visu"+str(rank)+"-"+str(miter)+".vtu",
                                  points, elements, cell_data=data, file_format="vtu")
    else:
        meshio.write_points_cells("results/visu"+str(rank)+"-"+str(miter)+".vtu",
                                  points, elements, point_data=data, file_format="vtu")

    if(rank == 0 and size > 1):
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
            if len(w_c) == len(cells):
                text_file.write("<PCellData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"v\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"c\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h+Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"exact\" format=\"binary\"/>\n")
                text_file.write("</PCellData>\n")
            else:
                text_file.write("<PPointData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"v\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"c\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"h+Z\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"Float64\" Name=\"exact\" format=\"binary\"/>\n")
                text_file.write("</PPointData>\n")
            for i in range(size):
                name1 = "visu"
                bu1 = [10]
                bu1 = str(i)
                name1 += bu1
                name1 += "-"+str(miter)
                name1 += ".vtu"
                text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
            text_file.write("</PUnstructuredGrid>\n")
            text_file.write("</VTKFile>")
