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
    flux.hc = 0
    flux.Z = 0

    return flux

@njit
def compute_flux_shallow_roe(flux, fleft, fright, w_l, w_r, normal):
    grav = 9.81

    mesn = np.sqrt(normal[0]*normal[0] + normal[1]*normal[1])
    norm = normal/mesn

    hroe = (w_l.h + w_r.h)/2
    uroe = (w_l.hu / w_l.h * np.sqrt(w_l.h)
            + w_r.hu / w_r.h * np.sqrt(w_r.h)) /(np.sqrt(w_l.h) + np.sqrt(w_r.h))

    vroe = (w_l.hv / w_l.h * np.sqrt(w_l.h)
            + w_r.hv / w_r.h * np.sqrt(w_r.h)) /(np.sqrt(w_l.h) + np.sqrt(w_r.h))

    croe = (w_l.hc / w_l.h * np.sqrt(w_l.h)
            + w_r.hc / w_r.h * np.sqrt(w_r.h)) /(np.sqrt(w_l.h) + np.sqrt(w_r.h))

    velson = np.sqrt(grav * hroe)

    lambda1 = uroe*norm[0] + vroe*norm[1] - velson
    lambda2 = uroe*norm[0] + vroe*norm[1]
    lambda3 = uroe*norm[0] + vroe*norm[1]
    lambda4 = uroe*norm[0] + vroe*norm[1] + velson

    rmat = np.zeros((4, 4))
    rmati = np.zeros((4, 4))
    almat = np.zeros((4, 4))
    ralmat = np.zeros((4, 4))
    ammat = np.zeros((4, 4))

    almat[0][0] = np.fabs(lambda1)
    almat[1][1] = np.fabs(lambda2)
    almat[2][2] = np.fabs(lambda3)
    almat[3][3] = np.fabs(lambda4)

    rmat[0][0] = 1.
    rmat[1][0] = uroe - norm[0]*velson
    rmat[2][0] = vroe - norm[1]*velson
    rmat[3][0] = croe
    rmat[3][1] = 1.
    rmat[1][2] = -1*norm[1]
    rmat[2][2] = norm[0]
    rmat[3][2] = 1
    rmat[0][3] = 1
    rmat[1][3] = uroe + norm[0]*velson
    rmat[2][3] = vroe + norm[1]*velson
    rmat[3][3] = croe

    rmati[0][0] = 0.5 *(1 + (uroe*norm[0] + vroe*norm[1])/velson)
    rmati[0][1] = -norm[0]/(2*velson)
    rmati[0][2] = -norm[1]/(2*velson)

    rmati[1][0] = vroe*norm[0] - uroe*norm[1] - croe
    rmati[1][1] = norm[1]
    rmati[1][2] = -1*norm[0]
    rmati[1][3] = 1

    rmati[2][0] = -vroe*norm[0] + uroe*norm[1]
    rmati[2][1] = -1*norm[1]
    rmati[2][2] = norm[0]

    rmati[3][0] = 0.5 *(1 - (uroe*norm[0] + vroe*norm[1])/velson)
    rmati[3][1] = norm[0]/(2*velson)
    rmati[3][2] = norm[1]/(2*velson)

    ralmat = matmul(ralmat, rmat, almat)
    ammat = matmul(ammat, ralmat, rmati)

    huleft = w_l.hu
    hvleft = w_l.hv
    huright = w_r.hu
    hvright = w_r.hv

    w_dif = np.zeros(4)
    w_dif[0] = w_r.h  - w_l.h
    w_dif[1] = huright - huleft
    w_dif[2] = hvright - hvleft
    w_dif[3] = w_r.hc - w_l.hc

    hnew = 0.
    unew = 0.
    vnew = 0.
    cnew = 0.

    for i in range(3):
        hnew += ammat[0][i] * w_dif[i]
        unew += ammat[1][i] * w_dif[i]
        vnew += ammat[2][i] * w_dif[i]
        cnew += ammat[3][i] * w_dif[i]


    u_h = hnew/2
    u_hu = unew/2
    u_hv = vnew/2
    u_hc = cnew/2


    q_l = w_l.hu * norm[0] + w_l.hv * norm[1]
    p_l = 0.5 * grav * w_l.h * w_l.h

    q_r = w_r.hu * norm[0] + w_r.hv * norm[1]
    p_r = 0.5 * grav * w_r.h * w_r.h

    fleft.h = q_l
    fleft.hu = q_l * w_l.hu/w_l.h + p_l*norm[0]
    fleft.hv = q_l * w_l.hv/w_l.h + p_l*norm[1]
    fleft.hc = q_l * w_l.hc/w_l.h

    fright.h = q_r
    fright.hu = q_r * w_r.hu/w_r.h + p_r*norm[0]
    fright.hv = q_r * w_r.hv/w_r.h + p_r*norm[1]
    fright.hc = q_r * w_r.hc/w_r.h


    f_h = 0.5 * (fleft.h + fright.h) - u_h
    f_hu = 0.5 * (fleft.hu + fright.hu) - u_hu
    f_hv = 0.5 * (fleft.hv + fright.hv) - u_hv
    f_hc = 0.5 * (fleft.hc + fright.hc) - u_hc

    flux.h = f_h * mesn
    flux.hu = f_hu * mesn
    flux.hv = f_hv * mesn
    flux.hc = f_hc * mesn
    flux.Z = 0

    return flux

@njit 
def term_source_vasquez(w_c, w_ghost, nodeidc, faceidc, cellidc, centerc, volumec, cellidf,
                        nodeidf, normalf, centerf, vertex, mystruct):
    grav = 9.81

    source = np.zeros(len(w_c), dtype=mystruct)
    trv = np.zeros(1, dtype=mystruct)[0]


    for i in range(len(w_c)):
        G = centerc[i]
        phi_ij = np.zeros(4)

        for j in range(3):
            f = faceidc[i][j]
            n1 = vertex[nodeidf[f][0]]
            n2 = vertex[nodeidf[f][1]]


#            A = n2[1] - n1[1]
#            B = n1[0] - n2[0]
#            C = -n1[0]*(n2[1]-n1[1]) + n1[1]*(n2[0]-n1[0])
#
#            d_ij_1 = np.fabs(A*G[0] + B*G[1]+C)/np.sqrt(A**2+B**2)

            c = centerf[f]

            mesn = np.sqrt(normalf[f][0]*normalf[f][0] + normalf[f][1]*normalf[f][1])
            aire_ij = 1./2 * np.fabs((n1[0] - n2[0])*(n1[1]-G[1]) - (n1[0] - G[0])*(n1[1]-n2[1]))
            dist_ij = 2 * aire_ij / mesn

            if cellidf[f][1] != -1:
                trv = w_c[cellidc[i][j]]
            else:
                trv = w_ghost[f]


            if np.dot(G-c, normalf[f]) < 0.0:
                ss = normalf[f]/mesn
            else:
                ss = -1.0*normalf[f]/mesn

            hroe = (w_c[i].h + trv.h)/2
            uroe = (w_c[i].hu / w_c[i].h * np.sqrt(w_c[i].h)
                    + trv.hu / trv.h * np.sqrt(trv.h)) /(np.sqrt(w_c[i].h) + np.sqrt(trv.h))

            vroe = (w_c[i].hv / w_c[i].h * np.sqrt(w_c[i].h)
                    + trv.hv / trv.h * np.sqrt(trv.h)) /(np.sqrt(w_c[i].h) + np.sqrt(trv.h))

            croe = (w_c[i].hc / w_c[i].h * np.sqrt(w_c[i].h)
                    + trv.hc / trv.h * np.sqrt(trv.h)) /(np.sqrt(w_c[i].h) + np.sqrt(trv.h))


            velson = np.sqrt(grav * hroe)

            lambda1 = uroe*ss[0] + vroe*ss[1] - velson
            lambda2 = uroe*ss[0] + vroe*ss[1]
            lambda3 = uroe*ss[0] + vroe*ss[1]
            lambda4 = uroe*ss[0] + vroe*ss[1] + velson

            rmat = np.zeros((4, 4))
            rmati = np.zeros((4, 4))
            almat = np.zeros((4, 4))
            ralmat = np.zeros((4, 4))
            ammat = np.zeros((4, 4))
            ident = np.zeros((4, 4))
            s_ij = np.zeros(4)


            if lambda1 > 1e-3:
                sign1 = 1
            elif lambda1 < -1e-3:
                sign1 = -1
            else:
                sign1 = 0

            if lambda2 > 1e-3:
                sign2 = 1
            elif lambda2 < -1e-03:
                sign2 = -1
            else:
                sign2 = 0

            if lambda3 > 1e-3:
                sign3 = 1
            elif lambda3 < -1e-03:
                sign3 = -1
            else:
                sign3 = 0

            if lambda4 > 1e-3:
                sign4 = 1
            elif lambda4 < -1e-03:
                sign4 = -1
            else:
                sign4 = 0

            ident[0][0] = 1.
            ident[1][1] = 1.
            ident[2][2] = 1.
            ident[3][3] = 1.

            almat[0][0] = sign1
            almat[1][1] = sign2
            almat[2][2] = sign3
            almat[3][3] = sign4

            rmat[0][0] = 1.
            rmat[1][0] = uroe - ss[0]*velson
            rmat[2][0] = vroe - ss[1]*velson
            rmat[3][0] = croe
            rmat[3][1] = 1.
            rmat[1][2] = -1*ss[1]
            rmat[2][2] = ss[0]
            rmat[3][2] = 1.
            rmat[0][3] = 1.
            rmat[1][3] = uroe + ss[0]*velson
            rmat[2][3] = vroe + ss[1]*velson
            rmat[3][3] = croe

            rmati[0][0] = 0.5 *(1 + (uroe*ss[0] + vroe*ss[1])/velson)
            rmati[0][1] = -ss[0]/(2*velson)
            rmati[0][2] = -ss[1]/(2*velson)

            rmati[1][0] = vroe*ss[0] - uroe*ss[1] - croe
            rmati[1][1] = ss[1]
            rmati[1][2] = -1*ss[0]
            rmati[1][3] = 1

            rmati[2][0] = -vroe*ss[0] + uroe*ss[1]
            rmati[2][1] = -1*ss[1]
            rmati[2][2] = ss[0]

            rmati[3][0] = 0.5 *(1 - (uroe*ss[0] + vroe*ss[1])/velson)
            rmati[3][1] = ss[0]/(2*velson)
            rmati[3][2] = ss[1]/(2*velson)

            ralmat = matmul(ralmat, rmat, almat)
            ammat = matmul(ammat, ralmat, rmati)

            matricer = ident - ammat

            s_ij[0] = 0
            s_ij[1] = -grav * (w_c[i].h + trv.h)/2 * (trv.Z - w_c[i].Z)/dist_ij * ss[0]
            s_ij[2] = -grav * (w_c[i].h + trv.h)/2 * (trv.Z - w_c[i].Z)/dist_ij * ss[1]
            s_ij[3] = 0

            for k in range(len(phi_ij)):
                phi_ij[0] += matricer[0][k] * s_ij[k]
                phi_ij[1] += matricer[1][k] * s_ij[k]
                phi_ij[2] += matricer[2][k] * s_ij[k]
                phi_ij[3] += matricer[3][k] * s_ij[k]

            source[i].h += phi_ij[0] * aire_ij
            source[i].hu += phi_ij[1] * aire_ij
            source[i].hv += phi_ij[2] * aire_ij
            source[i].hc += phi_ij[3] * aire_ij
            source[i].Z = 0

    return source

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
           + wr_l.hu / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wn_l.h) + np.sqrt(wr_l.h))

    v_h = (wn_l.hv / wn_l.h * np.sqrt(wn_l.h)
           + wr_l.hv / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wn_l.h) + np.sqrt(wr_l.h))

    c_h = (wn_l.hc / wn_l.h * np.sqrt(wn_l.h)
           + wr_l.hc / wr_l.h * np.sqrt(wr_l.h)) /(np.sqrt(wn_l.h) + np.sqrt(wr_l.h))


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
def term_source_srnh(w_c, w_ghost, w_halo, nodeidc, faceidc, cellidc, centerc, volumec, cellidf,
                     nodeidf, normalf, centerf, vertex, name, halofid, mystruct):

    source = np.zeros(len(w_c), dtype=mystruct)
    trv = np.zeros(1, dtype=mystruct)[0]

    zv = np.zeros(3)
    ss = np.zeros((3, 2))

    mata = np.zeros(3)
    matb = np.zeros(3)


    grav = 9.81

    for i in range(len(source)):
        ns = np.zeros((3, 2))

        G = centerc[i]
        c_1 = 0
        c_2 = 0

        for j in range(3):
            f = faceidc[i][j]
            c = centerf[f]

            if name[f] == 10:
                trv = w_halo[halofid[f]]

            elif name[f] == 0:# cellidf[f][1] != -1:
                trv = w_c[cellidc[i][j]]
            else:
                trv = w_ghost[f]

            if np.dot(G-c, normalf[f]) < 0.0:
                ss[j] = normalf[f]
            else:
                ss[j] = -1.0*normalf[f]

            zv[j] = trv.Z
            mata[j] = trv.h*ss[j][0]
            matb[j] = trv.h*ss[j][1]
            c_1 = c_1 + pow(0.5*(w_c[i].h+trv.h), 2)*ss[j][0]
            c_2 = c_2 + pow(0.5*(w_c[i].h+trv.h), 2)*ss[j][1]

        c_3 = 3.0 * w_c[i].h

        delta = (mata[1]*matb[2]-mata[2]*matb[1]) - (mata[0]*matb[2]-matb[0]*mata[2]) + (mata[0]*matb[1]-matb[0]*mata[1])

        deltax = c_3*(mata[1]*matb[2]-mata[2]*matb[1]) - (c_1*matb[2]-c_2*mata[2]) + (c_1*matb[1]-c_2*mata[1])

        deltay = (c_1*matb[2]-c_2*mata[2]) - c_3*(mata[0]*matb[2]-matb[0]*mata[2]) + (mata[0]*c_2-matb[0]*c_1)

        deltaz = (mata[1]*c_2-matb[1]*c_1) - (mata[0]*c_2-matb[0]*c_1) + c_3*(mata[0]*matb[1]-matb[0]*mata[1])

        h_1 = deltax/delta
        h_2 = deltay/delta
        h_3 = deltaz/delta

        z_1 = w_c[i].Z + w_c[i].h - h_1
        z_2 = w_c[i].Z + w_c[i].h - h_2
        z_3 = w_c[i].Z + w_c[i].h - h_3

        b = np.array([vertex[nodeidc[i][1]][0], vertex[nodeidc[i][1]][1]])

        ns[0] = np.array([(G[1]-b[1]), -(G[0]-b[0])])
        ns[1] = ns[0] - ss[1]  #  N23
        ns[2] = ns[0] + ss[0]  #  N31

#        s_1 = h_1*((z_1+zv[0])*(ss[0]/2.0) + (z_1+z_2)*(ns[0]/2.0) + (z_1+z_3)*((-1)*ns[2]/2.0));
#        s_2 = h_2*((z_2+zv[1])*(ss[1]/2.0) + (z_2+z_1)*((-1)*ns[0]/2.0) + (z_2+z_3)*(ns[1]/2.0));
#        s_3 = h_3*((z_3+zv[2])*(ss[2]/2.0) + (z_3+z_1)*(ns[2]/2.0) + (z_3+z_2)*((-1)*ns[1]/2.0));


        s_1 = 0.5*h_1*(zv[0]*ss[0] + z_2*ns[0] + z_3*(-1)*ns[2])
        s_2 = 0.5*h_2*(zv[1]*ss[1] + z_1*(-1)*ns[0] + z_3*ns[1])
        s_3 = 0.5*h_3*(zv[2]*ss[2] + z_1*ns[2] + z_2*(-1)*ns[1])

        source[i].h = 0
        source[i].hu = -grav*(s_1[0] + s_2[0] + s_3[0])
        source[i].hv = -grav*(s_1[1] + s_2[1] + s_3[1])
        source[i].hc = 0.
        source[i].Z = 0.


    return source

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
    flux.Z = 0.

    return flux
#
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

@njit
def barthlimiter(w_c, w_x, w_y, psi, cellid, faceid, centerc, centerf):
    var = "hu"
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
                   centerc, centerh, centerf, normal, halofid, name, mystruct, order):

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

    elif order == 2:

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

    grav = 9.81
    h_m = 2.534
    u_m = 4.03
    h_1 = 5.
    h_2 = 1.

    s = np.sqrt(43.925)

    x_0 = 6.
    x_1 = x_0 - time * np.sqrt(grav * h_1)
    x_2 = x_0 + (u_m - np.sqrt(grav * h_m))
    x_3 = x_0 + s*time

    for i in range(len(center)):
        xcent = center[i][0]
        if xcent < x_1:
            hexact[i] = h_1
            uexact[i] = 0

        elif xcent < x_2 and xcent >= x_1:
            hexact[i] = 1/(9*grav) * (2*np.sqrt(grav*h_1) - (xcent-x_0)/time)**2
            uexact[i] = 2/3 * (np.sqrt(grav*h_1) + (xcent-x_0)/time)

        elif xcent < x_3 and xcent >= x_2:
            hexact[i] = h_m
            uexact[i] = u_m

        elif xcent >= x_3:
            hexact[i] = h_2
            uexact[i] = 0


#        if np.fabs(xcent - 1500/2) <= 1500/8 :
#            zexact[i] = 8#10 + (40*xcent/14000) + 10*np.sin(np.pi*(4*xcent/14000 - 0.5))
#
#        hexact[i] = 20 - zexact[i] - 4*np.sin(np.pi*(4*time/86400 + 0.5))
#        #hexact[i] = 64.5 - zexact[i] - 4*np.sin(np.pi*(4*time/86400 + 0.5))
#
#        uexact[i] = (xcent - 1500)*np.pi/(5400*hexact[i]) * np.cos(np.pi*(4*time/86400 + 0.5))

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

    choix = 0 # (0,creneau 1:gaussienne)
    if choix == 0:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]
            if xcent <= 6:
                w_c.h[i] = 5.
            else:
                w_c.h[i] = 1.

            w_c.hu[i] = 0.
            w_c.hv[i] = 0.
            w_c.hc[i] = 0.
            w_c.Z[i] = 0.

    elif choix == 1:
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            w_c.h[i] = 1+np.exp(-30*(pow(xcent-2, 2) + pow(ycent-1.5, 2)))
            w_c.hc[i] = 0#np.exp(-30*(pow(xcent-2, 2) + pow(ycent-1.5, 2)))
#            w_c.hc[i] = c_1 * np.exp(-1* (pow(xcent - x_1, 2)
#                + pow(ycent -y_1, 2))/ pow(sigma1, 2)) + c_2 * np.exp(-1* (pow(xcent - x_2, 2)
#                + pow(ycent -y_2, 2))/ pow(sigma2, 2))
            w_c.hu[i] = 0.5
            w_c.hv[i] = 0.
            w_c.Z[i] = 0.

    elif choix == 2:
        L = 14000
        for i in range(nbelements):
            xcent = center[i][0]
            ycent = center[i][1]

            #w_c.Z[i] = 0.8 * np.exp(-5*(xcent - 1)**2 - 50* (ycent - 0.5)**2) #
            w_c.Z[i] = 10 + (40*xcent/L) + 10*np.sin(np.pi*(4*xcent/L - 0.5))
            #w_c.h[i] = 1 - w_c.Z[i]
            w_c.h[i] = 60.5 - w_c.Z[i]

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

#    L = 86400
    for i in range(len(cellid)):
#        w_ghost[i] = w_c[cellid[i][0]]
#        aa = 4*time/L + 0.5
#
#        if name[i] == 1:
#            w_ghost[i].h = 20 - 4*np.sin(np.pi * aa)
#            u = -1500 * np.pi/(5400*w_ghost[i].h) * np.cos(np.pi * aa)
#            v = 0
#
#            w_ghost[i].hu = w_ghost[i].h * u
#            w_ghost[i].hv = w_ghost[i].h * v
#
#        elif name[i] == 2:
#            w_ghost[i].hu = 0
#            w_ghost[i].hv = 0

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

        elif (name[i] == 1 or name[i] == 2):
            w_ghost[i] = w_c[cellid[i][0]]
#        if name[i] !=0:
#            w_ghost[i] = w_c[cellid[i][0]]

    return w_ghost

def save_paraview_results(w_c, solexact, niter, miter, time, dtime, rank, size, cells, nodes):

    elements = {"triangle": cells}
    points = []
    for i in nodes:
        points.append([i[0], i[1], i[2]])

    data = {"h" : w_c.h, "u" : w_c.hu/w_c.h, "v": w_c.hv/w_c.h, "c": w_c.hc/w_c.h, "Z": w_c.Z,
            "h+Z": w_c.h + w_c.Z, "exact": solexact}
    data = {"h": data, "u":data, "v": data, "c": data, "Z":data, "h+Z":data, "exact":data}

    maxh = np.zeros(1)
    maxh = max(w_c.h)
    integral_sum = np.zeros(1)

    COMM.Reduce(maxh, integral_sum, MPI.MAX, 0)
    if rank == 0:
        print(" **************************** Computing ****************************")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Iteration = ", niter, "time = ", np.float16(time), "time step = ", np.float16(dtime))
        print("max h =", np.float16(integral_sum[0]))

    meshio.write_points_cells("results/visu"+str(rank)+"-"+str(miter)+".vtu",
                              points, elements, cell_data=data, file_format="vtu")

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
            text_file.write("<PDataArray type=\"Float64\" Name=\"h+Z\" format=\"binary\"/>\n")
            text_file.write("<PDataArray type=\"Float64\" Name=\"exact\" format=\"binary\"/>\n")
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
