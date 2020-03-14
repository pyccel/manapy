#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:26:53 2020

@author: kissami
"""
import meshio
from mpi4py import MPI
import numpy as np
from numba import njit, jit


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

@njit
def compute_flux_advection(flux, fleft, fright, wleft, wright, normal):

    sol = 0
    vel = np.zeros(2)

    vel[0] = 0.5*(wleft.hu + wright.hu)
    vel[1] = 0.5*(wleft.hv + wright.hv)

    sign = np.dot(vel, normal)

    if sign >= 0:
        sol = wleft.h
    else:
        sol = wright.h

    flux.h = sign * sol
    flux.hu = 0
    flux.hv = 0

    return flux

@njit
def compute_flux_shallow_roe(flux, fleft, fright, wleft, wright, normal):
    grav = 9.81

    wnleft = wleft
    wnright = wright

    tinv = np.zeros(2)
    tinv[0] = -1*normal[1]
    tinv[1] = normal[0]
    mest = np.sqrt(tinv[0]**2 + tinv[1]**2)
    mesn = np.sqrt(normal[0]**2 + normal[1]**2)


    uleft = wnleft.hu*normal[0] + wnleft.hv*normal[1]
    uleft = uleft / mesn
    vleft = wnleft.hu*tinv[0] + wnleft.hv*tinv[1]
    vleft = vleft / mest
    #WL.hu = ul
    #WL.hv = vl

    uright = wnright.hu*normal[0] + wnright.hv*normal[1]
    uright = uright / mesn
    vright = wnright.hu*tinv[0] + wnright.hv*tinv[1]
    vright = vright / mest
    #WR.hu = ur
    #WR.hv = vr


    u_lrh = (wnleft.h  + wnright.h)/2
    u_lrhu = (uleft/wnleft.h + vright/wnright.h)/2
    u_lrhv = (uright/wnleft.h + vright/wnright.h)/2


    velson = np.sqrt(grav * u_lrh)

    lambda1 = u_lrhu - velson
    lambda2 = u_lrhu
    lambda3 = u_lrhu + velson


    rmat = np.zeros((3, 3))
    rmati = np.zeros((3, 3))
    almat = np.zeros((3, 3))
    ralmat = np.zeros((3, 3))
    ammat = np.zeros((3, 3))

    almat[0][0] = np.fabs(lambda1)
    almat[1][0] = 0.
    almat[2][0] = 0.
    almat[0][1] = 0.
    almat[1][1] = np.fabs(lambda2)
    almat[2][1] = 0.
    almat[0][2] = 0.
    almat[1][2] = 0.
    almat[2][2] = np.fabs(lambda3)

    rmat[0][0] = 1.
    rmat[1][0] = lambda1
    rmat[2][0] = u_lrhv
    rmat[0][1] = 0.
    rmat[1][1] = 0.
    rmat[2][1] = 1.
    rmat[0][2] = 1.
    rmat[1][2] = lambda3
    rmat[2][2] = u_lrhv

    rmati[0][0] = lambda3/(2*velson)
    rmati[1][0] = -u_lrhv
    rmati[2][0] = -lambda1/(2*velson)
    rmati[0][1] = -1./(2*velson)
    rmati[1][1] = 0.
    rmati[2][1] = 1./(2*velson)
    rmati[0][2] = 0.
    rmati[1][2] = 1.
    rmati[2][2] = 0.

    ralmat = matmul(ralmat, rmat, almat)
    ammat = matmul(ammat, ralmat, rmati)

    w_dif = np.zeros(3)
    w_dif[0] = wnright.h  - wnleft.h
    w_dif[1] = uright - uleft #WR.hu - WL.hu
    w_dif[2] = vright - vleft #WR.hv - WL.hv

    hnew = 0.
    unew = 0.
    vnew = 0.

    for i in range(3):
        hnew += ammat[0][i] * w_dif[i]
        unew += ammat[1][i] * w_dif[i]
        vnew += ammat[2][i] * w_dif[i]

    u_h = hnew/2
    u_hu = unew/2
    u_hv = vnew/2

    signl = uleft#WL.hu
    signr = uright#WR.hu

    p_gravl = 0.5 * grav * wnleft.h*wnleft.h
    p_gravr = 0.5 * grav * wnright.h*wnright.h

    fleft.h = signl
    fleft.hu = signl * uleft/wnleft.h + p_gravl #WL.hu/WL.h +pl
    fleft.hv = signl * vleft/wnleft.h      #WL.hv/WL.h


    fright.h = signr
    fright.hu = signr * uright/wnright.h + p_gravr#WR.hu/WR.h +pr
    fright.hv = signr * vright/wnright.h      #WR.hv/WR.h


    f_h = 0.5 * (fleft.h + fright.h) - u_h
    f_hu = 0.5 * (fleft.hu + fright.hu) - u_hu
    f_hv = 0.5 * (fleft.hv + fright.hv) - u_hv



    flux.h = f_h * mesn
    flux.hu = (f_hu*normal[0] + f_hv*-1*normal[1])
    flux.hv = (f_hu*-1*tinv[0] + f_hv*tinv[1])
    flux.hc = 0
    flux.Z = 0

    #print(flux)


    return flux

@njit(fastmath=True)
def compute_flux_shallow_srnh(flux, fleft, fright, w_l, w_r, normal):
    grav = 9.81

    wn_l = w_l
    wr_l = w_r

    ninv = np.zeros(2)
    ninv[0] = -1*normal[1]
    ninv[1] = normal[0]
    mesninv = np.sqrt(ninv[0]*ninv[0] + ninv[1]*ninv[1])
    mesn = np.sqrt(normal[0]*normal[0] + normal[1]*normal[1])


    u_h = (wn_l.hu / wn_l.h * np.sqrt(wn_l.h)
           + wr_l.hu / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wr_l.h) + np.sqrt(wr_l.h))
    v_h = (wn_l.hv / wn_l.h * np.sqrt(wn_l.h)
           + wr_l.hv / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wr_l.h) + np.sqrt(wr_l.h))
    c_h = (wn_l.hc / wn_l.h * np.sqrt(wn_l.h)
           + wr_l.hc / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wr_l.h) + np.sqrt(wr_l.h))


    #uvh = np.array([uh, vh])
    un_h = u_h*normal[0] + v_h*normal[1]#np.dot(uvh , n);
    un_h = un_h / mesn
    vn_h = u_h*ninv[0] + v_h*ninv[1]#np.dot(uvh , t);
    vn_h = vn_h / mesninv

    hroe = (wn_l.h+wr_l.h)/2
    uroe = un_h
    vroe = vn_h
    croe = c_h

    uleft = wn_l.hu*normal[0] + wn_l.hv*normal[1]
    uleft = uleft / mesn
    vleft = wn_l.hu*ninv[0] + wn_l.hv*ninv[1]
    vleft = vleft / mesninv

    uright = wr_l.hu*normal[0] + wr_l.hv*normal[1]
    uright = uright / mesn
    vright = wr_l.hu*ninv[0] + wr_l.hv*ninv[1]
    vright = vright / mesninv



    w_lrh = (wn_l.h  + wr_l.h)/2
    w_lrhu = (uleft + uright)/2
    w_lrhv = (vleft + vright)/2
    w_lrhc = (wn_l.hc  + wr_l.hc)/2
    w_lrz = (wn_l.Z + wr_l.Z)/2


    sound = np.sqrt(grav * hroe)

    lambda1 = uroe - sound
    #lambda2 = 0
    lambda3 = uroe
    lambda4 = uroe
    lambda5 = uroe + sound

    if lambda1 == 0:
        sign1 = 0.
    else:
        sign1 = lambda1 / np.fabs(lambda1)

    sign2 = 0.

    if lambda3 == 0:
        sign3 = 0.
    else:
        sign3 = lambda3 / np.fabs(lambda3)

    if lambda4 == 0:
        sign4 = 0.
    else:
        sign4 = lambda4 / np.fabs(lambda4)

    if lambda5 == 0:
        sign5 = 0.
    else:
        sign5 = lambda5 / np.fabs(lambda5)

    rmat = np.zeros((5, 5))
    rmati = np.zeros((5, 5))
    slmat = np.zeros((5, 5))
    rslmat = np.zeros((5, 5))
    smmat = np.zeros((5, 5))

    slmat[0][0] = sign1
    slmat[1][1] = sign2
    slmat[2][2] = sign3
    slmat[3][3] = sign4
    slmat[4][4] = sign5

    rmat[0][0] = 1.
    rmat[1][0] = lambda1 #(u-c)
    rmat[2][0] = vroe
    rmat[3][0] = croe

    rmat[0][1] = 1.
    rmat[2][1] = vroe
    rmat[3][1] = croe
    rmat[4][1] = (uroe**2 - sound**2)/(sound**2)

    rmat[2][2] = 1.

    rmat[3][3] = 1.

    rmat[0][4] = 1.
    rmat[1][4] = lambda5#(u+c)
    rmat[2][4] = vroe
    rmat[3][4] = croe


    rmati[0][0] = lambda5/(2*sound)
    rmati[2][0] = -vroe
    rmati[3][0] = -croe
    rmati[4][0] = -lambda1/(2*sound)

    rmati[0][1] = -1./(2*sound)
    rmati[4][1] = 1./(2*sound)

    rmati[2][2] = 1.
    rmati[3][3] = 1.

    rmati[0][4] = -sound/(2*lambda1)
    rmati[1][4] = sound**2/(uroe**2 - sound**2)
    rmati[4][4] = sound/(2*lambda5)

    rslmat = matmul(rslmat, rmat, slmat)
    smmat = matmul(smmat, rslmat, rmati)

    w_dif = np.zeros(5)
    w_dif[0] = wr_l.h  - wn_l.h
    w_dif[1] = uright - uleft
    w_dif[2] = vright - vleft
    w_dif[3] = wr_l.hc - wn_l.hc
    w_dif[4] = wr_l.Z - wn_l.Z

    hnew = 0.
    unew = 0.
    vnew = 0.
    cnew = 0.
    znew = 0.

    for i in range(5):
        hnew += smmat[0][i] * w_dif[i]
        unew += smmat[1][i] * w_dif[i]
        vnew += smmat[2][i] * w_dif[i]
        cnew += smmat[3][i] * w_dif[i]
        znew += smmat[4][i] * w_dif[i]

    u_h = hnew/2
    u_hu = unew/2
    u_hv = vnew/2
    u_hc = cnew/2
    u_z = znew/2

    w_lrh = w_lrh  - u_h
    w_lrhu = w_lrhu - u_hu
    w_lrhv = w_lrhv - u_hv
    w_lrhc = w_lrhc - u_hc
    w_lrz = w_lrz  - u_z

    unew = 0.
    vnew = 0.

    unew = w_lrhu * normal[0] + w_lrhv * -1*normal[1]
    unew = unew / mesn
    vnew = w_lrhu * -1*ninv[0] + w_lrhv * ninv[1]
    vnew = vnew / mesninv


    w_lrhu = unew
    w_lrhv = vnew

    q_s = normal[0] * unew + normal[1] * vnew


    flux.h = q_s
    flux.hu = q_s * w_lrhu/w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[0]
    flux.hv = q_s * w_lrhv/w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[1]
    flux.hc = q_s * w_lrhc/w_lrh
    flux.Z = 0.

    return flux

@njit(fastmath=True)
def compute_flux_shallow_rusanov(flux, fleft, fright, w_l, w_r, normal):

    grav = 9.81

    q_l = w_l.hu * normal[0] + w_l.hv * normal[1]
    p_l = 0.5 * grav * w_l.h * w_l.h

    mes = np.sqrt(normal[0]*normal[0] + normal[1]*normal[1])

    q_r = w_r.hu * normal[0] + w_r.hv * normal[1]
    p_r = 0.5 * grav * w_r.h * w_r.h

    fleft.h = q_l
    fleft.hu = q_l * w_l.hu/w_l.h + p_l*normal[0]
    fleft.hv = q_l * w_l.hv/w_l.h + p_l*normal[1]
    fleft.hc = q_l * w_l.hc/w_l.h

    fright.h = q_r
    fright.hu = q_r * w_r.hu/w_r.h + p_r*normal[0]
    fright.hv = q_r * w_r.hv/w_r.h + p_r*normal[1]
    fright.hc = q_r * w_r.hc/w_r.h

    c_l = np.sqrt(grav * w_l.h)
    c_r = np.sqrt(grav * w_r.h)

    f_l = np.fabs((q_l/mes)/w_l.h) + c_l
    f_r = np.fabs((q_r/mes)/w_r.h) + c_r

    if f_l > f_r:
        s_lr = f_l
    else:
        s_lr = f_r

    flux.h = 0.5 * (fleft.h + fright.h) - 0.5 * s_lr * mes * (w_r.h - w_l.h)
    flux.hu = 0.5 * (fleft.hu + fright.hu) - 0.5 * s_lr * mes * (w_r.hu - w_l.hu)
    flux.hv = 0.5 * (fleft.hv + fright.hv) - 0.5 * s_lr * mes * (w_r.hv - w_l.hv)
    flux.hc = 0.5 * (fleft.hc + fright.hc) - 0.5 * s_lr * mes * (w_r.hc - w_l.hc)
    flux.Z = 0

    return flux
#
@njit(fastmath=True)
def add(sol_1, sol_2):
    sol_1.h += sol_2.h
    sol_1.hu += sol_2.hu
    sol_1.hv += sol_2.hv
    sol_1.hc += sol_2.hc

    return sol_1

@njit(fastmath=True)
def minus(sol_1, sol_2):
    sol_1.h -= sol_2.h
    sol_1.hu -= sol_2.hu
    sol_1.hv -= sol_2.hv
    sol_1.hc -= sol_2.hc

    return sol_1

@jit(fastmath=True)#nopython = False)
def matmul(rmatrix, matrix1, matrix2):

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]
    return rmatrix

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

        dia = i_xx * i_yy - pow(i_xy, 2)

        w_x[i].h = (i_yy * j_xh - i_xy * j_yh) / dia
        w_y[i].h = (i_xx * j_yh - i_xy * j_xh) / dia

        w_x[i].hu = (i_yy * j_xhu - i_xy * j_yhu) / dia
        w_y[i].hu = (i_xx * j_yhu - i_xy * j_xhu) / dia

        w_x[i].hv = (i_yy * j_xhv - i_xy * j_yhv) / dia
        w_y[i].hv = (i_xx * j_yhv - i_xy * j_xhv) / dia

        w_x[i].hc = (i_yy * j_xhc - i_xy * j_yhc) / dia
        w_y[i].hc = (i_xx * j_yhc - i_xy * j_xhc) / dia

        w_x[i].Z = 0#(i_yy * j_xhc - i_xy * j_yhc) / D
        w_y[i].Z = 0#(i_xx * j_yhc - i_xy * j_xhc) / D

    return w_x, w_y

@njit
def barthlimiter(w_c, w_x, w_y, psi, cellid, faceid, centerc, centerf):
    var = "hc"
    for i in range(len(w_c)): psi[i] = 1

    for i in range(len(w_c)):
        w_max = w_c[var][i]
        w_min = w_c[var][i]

        for j in range(3):
            face = faceid[i][j]
            if cellid[face][1] != -1:
                w_max = max(w_max, w_c[var][cellid[face][0]], w_c[var][cellid[face][1]])
                w_min = min(w_min, w_c[var][cellid[face][0]], w_c[var][cellid[face][1]])
            else:
                w_max = max(w_max, w_c[var][cellid[face][0]])
                w_min = min(w_min, w_c[var][cellid[face][0]])

        for j in range(3):
            face = faceid[i][j]

            if cellid[face][1] != -1:
                r_xy = np.array([centerf[face][0] - centerc[i][0],
                                 centerf[face][1] - centerc[i][1]])
                delta2 = w_x[var][i] * r_xy[0] + w_y[var][i] * r_xy[1]

                if np.fabs(delta2) < 1e-8:
                    psi_ij = 1.
                else:
                    if delta2 > 0.:
                        value = (w_max - w_c[var][i]) / delta2
                        psi_ij = min(1., value)
                    if delta2 < 0.:
                        value = (w_min - w_c[var][i]) / delta2
                        psi_ij = min(1., value)

                psi[i] = min(psi[i], psi_ij)
            else:
                psi[i] = 1

    return psi

@njit(fastmath=True)
def albada(wleft, wright, w_x, w_y, center_left, center_right, lim):

    var_a = 0.
    var_b = 0.
    omega = 2/3
    epsilon = 0.
    limit = 0

    var_t = np.array([(center_right[0] - center_left[0]), (center_right[1] - center_left[1])])

    var_h = np.array([w_x.h, w_y.h])
    var_a = omega * np.dot(var_h, var_t) + (1-omega) * (wright.h - wleft.h)
    var_b = wright.h - wleft.h
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.h = limit

    var_hu = np.array([w_x.hu, w_y.hu])
    var_a = omega * np.dot(var_hu, var_t) + (1-omega) * (wright.hu - wleft.hu)
    var_b = wright.hu - wleft.hu
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.hu = limit

    var_hv = np.array([w_x.hv, w_y.hv])
    var_a = omega * np.dot(var_hv, var_t) + (1-omega) * (wright.hv - wleft.hv)
    var_b = wright.hv - wleft.hv
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.hv = limit

    var_hc = np.array([w_x.hc, w_y.hc])
    var_a = omega * np.dot(var_hc, var_t) + (1-omega) * (wright.hc - wleft.hc)
    var_b = wright.hc - wleft.hc
    if (var_a*var_b) > 0.:
        limit = ((var_a**2 + epsilon)*var_b + (var_b**2 + epsilon)*var_a) / (var_a**2 + var_b**2)
    else:
        limit = 0.

    lim.hc = limit

    lim.Z = 0.

    return lim

@njit(fastmath=True)
def explicitscheme(w_c, w_x, w_y, w_ghost, w_halo, wx_halo, wy_halo, cellid, faceid,
                   centerc, centerh, centerf, normal, halofid, name, mystruct):

    order = 1

    rezidus = np.zeros(len(w_c), dtype=mystruct)
    w_l = np.zeros(1, dtype=mystruct)[0]
    w_r = np.zeros(1, dtype=mystruct)[0]
    w_ln = np.zeros(1, dtype=mystruct)[0]
    w_rn = np.zeros(1, dtype=mystruct)[0]

    flx = np.zeros(1, dtype=mystruct)[0]
    fleft = np.zeros(1, dtype=mystruct)[0]
    fright = np.zeros(1, dtype=mystruct)[0]

    if order == 1:
        for i in range(len(cellid)):
            w_l = w_c[cellid[i][0]]
            norm = normal[i]

            if name[i] == 0:
                w_r = w_c[cellid[i][1]]

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm)

                rezidus[cellid[i][0]] = minus(rezidus[cellid[i][0]], flx)
                rezidus[cellid[i][1]] = add(rezidus[cellid[i][1]], flx)

            elif name[i] == 10:
                w_r = w_halo[halofid[i]]

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm)
                rezidus[cellid[i][0]] = minus(rezidus[cellid[i][0]], flx)

            else:
                w_r = w_ghost[i]

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm)
                rezidus[cellid[i][0]] = minus(rezidus[cellid[i][0]], flx)

    elif order == 3:

        psi = np.zeros(len(w_c))
        psi = barthlimiter(w_c, w_x, w_y, psi, cellid, faceid, centerc, centerf)

        for i in range(len(cellid)):
            w_l = w_c[cellid[i][0]]
            norm = normal[i]

            if name[i] == 0:
                w_r = w_c[cellid[i][1]]

                center_left = centerc[cellid[i][0]]
                center_right = centerc[cellid[i][1]]

                w_x_left = w_x[cellid[i][0]]
                w_y_left = w_y[cellid[i][0]]
                psi_left = psi[cellid[i][0]]

                w_x_right = w_x[cellid[i][1]]
                w_y_right = w_y[cellid[i][1]]
                psi_right = psi[cellid[i][1]]

                r_l = np.array([centerf[i][0] - center_left[0], centerf[i][1] - center_left[1]])
                w_ln.h = w_l.h  + psi_left * (w_x_left.h  * r_l[0] + w_y_left.h  * r_l[1])
                w_ln.hu = w_l.hu + psi_left * (w_x_left.hu * r_l[0] + w_y_left.hu * r_l[1])
                w_ln.hv = w_l.hv + psi_left * (w_x_left.hv * r_l[0] + w_y_left.hv * r_l[1])
                w_ln.hc = w_l.hc + psi_left * (w_x_left.hc * r_l[0] + w_y_left.hc * r_l[1])
                w_ln.Z = w_l.Z  + psi_left * (w_x_left.Z  * r_l[0] + w_y_left.Z  * r_l[1])

                r_r = np.array([centerf[i][0] - center_right[0], centerf[i][1] - center_right[1]])
                w_rn.h = w_r.h  + psi_right * (w_x_right.h  * r_r[0] + w_y_right.h  * r_r[1])
                w_rn.hu = w_r.hu + psi_right * (w_x_right.hu * r_r[0] + w_y_right.hu * r_r[1])
                w_rn.hv = w_r.hv + psi_right * (w_x_right.hv * r_r[0] + w_y_right.hv * r_r[1])
                w_rn.hc = w_r.hc + psi_right * (w_x_right.hc * r_r[0] + w_y_right.hc * r_r[1])
                w_rn.Z = w_r.Z  + psi_right * (w_x_right.Z  * r_r[0] + w_y_right.Z  * r_r[1])

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_ln, w_rn, norm)

                rezidus[cellid[i][0]] = minus(rezidus[cellid[i][0]], flx)
                rezidus[cellid[i][1]] = add(rezidus[cellid[i][1]], flx)

            elif name[i] == 10 and SIZE > 1:

                w_l = w_c[cellid[i][0]]
                w_r = w_halo[halofid[i]]

                center_left = centerc[cellid[i][0]]
                center_right = centerh[halofid[i]]

                w_x_left = w_x[cellid[i][0]]
                w_y_left = w_y[cellid[i][0]]
                psi_left = psi[cellid[i][0]]

                w_x_right = wx_halo[halofid[i]]
                w_y_right = wy_halo[halofid[i]]
                psi_right = psi[cellid[i][0]]

                r_l = np.array([centerf[i][0] - center_left[0], centerf[i][1] - center_left[1]])
                w_ln.h = w_l.h  + psi_left * (w_x_left.h  * r_l[0] + w_y_left.h  * r_l[1])
                w_ln.hu = w_l.hu + psi_left * (w_x_left.hu * r_l[0] + w_y_left.hu * r_l[1])
                w_ln.hv = w_l.hv + psi_left * (w_x_left.hv * r_l[0] + w_y_left.hv * r_l[1])
                w_ln.hc = w_l.hc + psi_left * (w_x_left.hc * r_l[0] + w_y_left.hc * r_l[1])
                w_ln.Z = w_l.Z  + psi_left * (w_x_left.Z  * r_l[0] + w_y_left.Z  * r_l[1])

                r_r = np.array([centerf[i][0] - centerh[halofid[i]][0],
                                centerf[i][1] - centerh[halofid[i]][1]])
                w_rn.h = w_r.h  + psi_right * (w_x_right.h  * r_r[0] + w_y_right.h  * r_r[1])
                w_rn.hu = w_r.hu + psi_right * (w_x_right.hu * r_r[0] + w_y_right.hu * r_r[1])
                w_rn.hv = w_r.hv + psi_right * (w_x_right.hv * r_r[0] + w_y_right.hv * r_r[1])
                w_rn.hc = w_r.hc + psi_right * (w_x_right.hc * r_r[0] + w_y_right.hc * r_r[1])
                w_rn.Z = w_r.Z  + psi_right * (w_x_right.Z  * r_r[0] + w_y_right.Z  * r_r[1])

                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_ln, w_rn, norm)
                rezidus[cellid[i][0]] = minus(rezidus[cellid[i][0]], flx)

            else:
                w_l = w_c[cellid[i][0]]
                w_r = w_ghost[i]
                flx = compute_flux_shallow_srnh(flx, fleft, fright, w_l, w_r, norm)
                rezidus[cellid[i][0]] = minus(rezidus[cellid[i][0]], flx)

    return rezidus

@njit(fastmath=True)
def update(w_c, wnew, dtime, rezidus, volume):
    for i in range(len(w_c)):
        wnew.h[i] = w_c.h[i]  + dtime * (rezidus["h"][i]/volume[i])
        wnew.hu[i] = w_c.hu[i] + dtime * (rezidus["hu"][i]/volume[i])
        wnew.hv[i] = w_c.hv[i] + dtime * (rezidus["hv"][i]/volume[i])
        wnew.hc[i] = w_c.hc[i] + dtime * (rezidus["hc"][i]/volume[i])
        wnew.Z[i] = w_c.Z[i]  + dtime * (rezidus["Z"][i]/volume[i])

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
def initialisation(w_c, center):

    nbelements = len(center)
    sigma1 = 264
    sigma2 = 264
    c_1 = 10
    c_2 = 6.5
    x_1 = 1400
    y_1 = 1400
    x_2 = 2400
    y_2 = 2400

    choix = 1 # (0,creneau 1:gaussienne)
    if choix == 0:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            if xcent <= 5.:
                w_c.h[i] = 5.
            else:
                w_c.h[i] = 2.5

            w_c.hu[i] = 0.
            w_c.hv[i] = 0.
            w_c.hc[i] = 1.
            w_c.Z[i] = 0.

    elif choix == 1:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            w_c.h[i] = 1#+np.exp(-30*(pow(x-2, 2) + pow(y-1.5, 2)))
            #w_c.hc[i] = np.exp(-30*(pow(xcent-2, 2) + pow(ycent-1.5, 2)))
            w_c.hc[i] = c_1 * np.exp(-1* (pow(xcent - x_1, 2) 
                + pow(ycent -y_1, 2))/ pow(sigma1, 2)) + c_2 * np.exp(-1* (pow(xcent - x_2, 2) 
                + pow(ycent -y_2, 2))/ pow(sigma2, 2))
            w_c.hu[i] = 0.5
            w_c.hv[i] = 0.5
            w_c.Z[i] = 0.

    return w_c

@njit
def ghost_value(w_c, w_ghost, cellid, name, normal):

    for i in range(len(cellid)):
#        if (name[i] == 3 or name[i] == 4 ):
#            #slip conditions
#            norm = normal[i]
#            #print(name[i], i)
#
#            u_i = w_c[cellid[i][0]].hu/w_c[cellid[i][0]].h
#            v_i = w_c[cellid[i][0]].hv/w_c[cellid[i][0]].h
#
#            mesn = np.sqrt(norm[0]*norm[0]+ norm[1]*norm[1])
#
#            s_n = norm / mesn
#
#            u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
#            v_g = v_i*(s_n[0]*s_n[0] - s_n[1]*s_n[1]) - 2.0*u_i*s_n[0]*s_n[1]
#
#            w_ghost[i].h = w_c[cellid[i][0]].h
#            w_ghost[i].Z = w_c[cellid[i][0]].Z
#            w_ghost[i].hc = w_c[cellid[i][0]].hc
#
#            w_ghost[i].hu = w_c[cellid[i][0]].h * u_g
#            w_ghost[i].hv = w_c[cellid[i][0]].h * v_g
#
#        elif name[i] != 0:
        w_ghost[i] = w_c[cellid[i][0]]

    return w_ghost

def save_paraview_results(w_c, niter, miter, time, dtime, rank, size, cells, nodes):

    elements = {"triangle": cells}
    points = []
    for i in nodes:
        points.append([i[0], i[1], i[2]])

    data = {"h" : w_c.h, "u" : w_c.hu/w_c.h, "v": w_c.hv/w_c.h, "c": w_c.hc/w_c.h, "Z": w_c.Z}
    data = {"h": data, "u":data, "v": data, "c": data, "Z":data}
    maxh = np.zeros(1)
    maxh = max(w_c.hc)
    integral_sum = np.zeros(1)

    print(type(data))
    COMM.Reduce(maxh, integral_sum, MPI.MAX, 0)
    if rank == 0:
        print(" **************************** Computing ****************************")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Iteration = ", niter, "time = ", np.float16(time), "time step = ", np.float16(dtime))
        print("max h =", np.float16(integral_sum[0]))

#    meshio.write_points_cells("results/visu"+str(rank)+"-"+str(miter)+".vtu",
#                              points, elements, cell_data=data, file_format="vtu")

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
            text_file.write("<PCellData Scalars=\"h\">\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"h\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"u\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"v\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"c\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"Z\" format=\"binary\"/>\n")
            text_file.write("</PCellData>\n")
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
