
def update_rhs_glob(ne:'float[:]', ni:'float[:]', loctoglob:'int[:]', rhs_updated:'float[:]'):
    
    for i in range(len(ne)):
        rhs_updated[loctoglob[i]] = 1.8096e-8 * (ne[i] - ni[i])
        
def update_rhs_loc(ne:'float[:]', ni:'float[:]', loctoglob:'int[:]', rhs_updated:'float[:]'):
    
    #$ omp parallel for 
    for i in range(len(ne)):
        rhs_updated[i] = 1.8096e-08 * (ne[i] - ni[i])
        
def initialisation_streamer(ne:'float[:]', ni:'float[:]', u:'float[:]', v:'float[:]', Ex:'float[:]', Ey:'float[:]', 
                            P:'float[:]', center:'float[:,:]', Pinit:'float'):
    from numpy import exp    
    nbelements = len(center)
    
    sigma = 0.01
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        
        ne[i] = 1e16 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2) / sigma**2) + 1e09
        ni[i] = 1e16 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2) / sigma**2) + 1e09

        u[i]  = 0.
        v[i]  = 0.
        P[i]  = Pinit * (1. - xcent)
        Ex[i] = 0.
        Ey[i] = 0.

      
def initialisation_streamer_3d(ne:'float[:]', ni:'float[:]', u:'float[:]', v:'float[:]', w:'float[:]', Ex:'float[:]', Ey:'float[:]', 
                                Ez:'float[:]', P:'float[:]', center:'float[:,:]', Pinit:'float'):
    
    from numpy import exp
    
    nbelements = len(center)
    
    sigma = 0.01
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        zcent = center[i][2]
        
        ne[i] = 1e16 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2 + (zcent-0.25)**2) / sigma**2) + 1e12
        ni[i] = 1e16 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2 + (zcent-0.25)**2) / sigma**2) + 1e12

        u[i] = 0.
        v[i] = 0.
        w[i] = 0.
        P[i] =  Pinit * (.5 - xcent)
        Ex[i] = 0.
        Ey[i] = 0.
        Ez[i] = 0.

def explicitscheme_dissipative_streamer(u_face:'float[:]', v_face:'float[:]', w_face:'float[:]',Ex_face:'float[:]',
                                        Ey_face:'float[:]', Ez_face:'float[:]', nex_face:'float[:]', ney_face:'float[:]', 
                                        nez_face:'float[:]', cellidf:'int[:,:]', normalf:'float[:,:]', namef:'int[:]', 
                                        dissip_ne:'float[:]'):
    
    from numpy import sqrt, zeros

    nbface = len(cellidf)
    norm = zeros(3)
    dissip_ne[:] = 0.
    
    for i in range(nbface):
        
        norm[:] = normalf[i][:]
        
        q = nex_face[i] * norm[0] + ney_face[i] * norm[1] + nez_face[i] * norm[2]
        n = 2.5e19;
        E = sqrt(Ex_face[i]**2 + Ey_face[i]**2 + Ez_face[i]**2)
        ve = sqrt(u_face[i]**2 + v_face[i]**2  + w_face[i]**2)
        De = (0.3341e9 * (E/n)**0.54069) * (ve/E)
        
        if (E == 0.): 
            De = 0.
        flux_ne = De*q

        if namef[i] == 0:
            dissip_ne[cellidf[i][0]] += flux_ne
            dissip_ne[cellidf[i][1]] -= flux_ne
        else:
            dissip_ne[cellidf[i][0]] += flux_ne
            
            
def explicitscheme_source_streamer(ne:'float[:]', u:'float[:]', v:'float[:]', w:'float[:]', Ex:'float[:]', Ey:'float[:]', 
                                    Ez:'float[:]', src_ne:'float[:]', src_ni:'float[:]',  center:'float[:,:]',  br:'int'):

    from numpy import sqrt, exp
    n = 2.5e19
    nbelements = len(ne)
    
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        zcent = center[i][2]
        
        E = sqrt(Ex[i]**2 + Ey[i]**2 + Ez[i]**2 )
        ve = sqrt(u[i]**2 + v[i]**2 + w[i]**2 )
        alpha_n =0.
        
        if ((E/n) > 1.5e-15):
            alpha_n = 2e-16 * exp(-7.248e-15/(E/n))
        
        else:
              alpha_n = 6.619e-17 * exp(-5.593e-15/(E/n));
    
        S = alpha_n * ve * ne[i] * n
        
        if (br == 1):
            S += 1e25 * exp(-1.*((xcent-0.3)**2. + (ycent-0.25)**2. + (zcent-0.28)**2.) / (0.005**2.))

        if (br == 2):
            S += 1e25 * exp(-1.*((xcent-0.31)**2. + (ycent-0.25)**2. + (zcent-0.22)**2.) / (0.005**2.))
                    
        src_ne[i] = S
        src_ni[i] = S

def compute_el_field(Pgradfacex:'float[:]', Pgradfacey:'float[:]', Pgradfacez:'float[:]', 
                     Ex_face:'float[:]', Ey_face:'float[:]', Ez_face:'float[:]'):
    
    #$ omp parallel for 
    for i in range(len(Ex_face)):
        Ex_face[i] = Pgradfacex[i]
        Ey_face[i] = Pgradfacey[i]
        Ez_face[i] = Pgradfacez[i]
        
        
def compute_velocity(Ex_face:'float[:]', Ey_face:'float[:]', Ez_face:'float[:]', u_face:'float[:]', v_face:'float[:]', 
                      w_face:'float[:]',  Ex:'float[:]', Ey:'float[:]',  Ez:'float[:]', u:'float[:]', v:'float[:]',
                      w:'float[:]', faceidc:'int[:,:]', dim:'int'):
    
    from numpy import sqrt
    
    n = 2.5e19
    nbfaces = len(u_face)
    nbelements = len(faceidc)
    u[:] = 0.; v[:] = 0.; w[:] = 0.; Ex[:] = 0.; Ey[:] = 0.; Ez[:] = 0.
    
    for i in range(nbfaces):
        E = sqrt(Ex_face[i]**2 + Ey_face[i]**2 + Ez_face[i]**2)
    
        if (E == 0.):
            u_face[i] = 0.
            v_face[i] = 0.
            w_face[i] = 0.
        
        elif ((E/n) > 2e-15):
            u_face[i] = -1./E *(7.4e21 * E/n + 7.1e6) * Ex_face[i]
            v_face[i] = -1./E *(7.4e21 * E/n + 7.1e6) * Ey_face[i]
            w_face[i] = -1./E *(7.4e21 * E/n + 7.1e6) * Ez_face[i]
            
        elif ((1e-16 < (E/n)) and ((E/n) <= 2e-15)):
            u_face[i] = -1./E * (1.03e22 * E/n + 1.3e6) * Ex_face[i]
            v_face[i] = -1./E * (1.03e22 * E/n + 1.3e6) * Ey_face[i]
            w_face[i] = -1./E * (1.03e22 * E/n + 1.3e6) * Ez_face[i]
            
        elif ((2.6e-17 < (E/n)) and ((E/n) <= 1e-16)):
            u_face[i] = -1./E * (7.2973e21 * E/n + 1.63e6) * Ex_face[i]
            v_face[i] = -1./E * (7.2973e21 * E/n + 1.63e6) * Ey_face[i]
            w_face[i] = -1./E * (7.2973e21 * E/n + 1.63e6) * Ez_face[i]
            
        elif ((E/n) <= 2.6e-17):
            u_face[i] = -1./E * (6.87e22 * E/n + 3.38e4) * Ex_face[i]
            v_face[i] = -1./E * (6.87e22 * E/n + 3.38e4) * Ey_face[i]
            w_face[i] = -1./E * (6.87e22 * E/n + 3.38e4) * Ez_face[i]
            
    for i in range(nbelements):
        for j in range(dim+1):
            u[i]  += u_face[faceidc[i][j]]
            v[i]  += v_face[faceidc[i][j]]
            w[i]  += w_face[faceidc[i][j]]
            
            Ex[i] += Ex_face[faceidc[i][j]]
            Ey[i] += Ey_face[faceidc[i][j]]
            Ez[i] += Ez_face[faceidc[i][j]]
            
        u[i]  /= (dim+1)
        v[i]  /= (dim+1)
        w[i]  /= (dim+1)
        
        Ex[i] /= (dim+1)
        Ey[i] /= (dim+1)
        Ez[i] /= (dim+1)
        
def time_step_streamer(u:'float[:]', v:'float[:]', w:'float[:]', Ex:'float[:]', Ey:'float[:]', Ez:'float[:]',
                        cfl:'float', normal:'float[:,:]', mesure:'float[:]', volume:'float[:]', 
                        faceid:'int[:,:]', dim:'int'):
    
    from numpy import zeros, sqrt, fabs
    
    nbelement =  len(faceid)
    u_n = 0.
    n = 2.5e19
    norm = zeros(3)
    dt = 1e6
  
    for i in range(nbelement):
        ve = sqrt(u[i]**2 + v[i]**2 + w[i]**2)
        E  = sqrt(Ex[i]**2 + Ey[i]**2 + Ez[i]**2)
      
        De = 0.3341e9 * (E/n)**0.54069 * (ve/E)
        lam = 0.
        
        for j in range(dim+1):
            norm[:] = normal[faceid[i][j]][:]
            
            #convective part
            u_n = fabs(u[i]*norm[0] + v[i]*norm[1] + w[i]*norm[2])
            lam_convect = u_n/mesure[faceid[i][j]] 
            lam += lam_convect * mesure[faceid[i][j]]
 
            #diffusion part
            mes = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2])
            lam_diff = De * mes**2
            lam += lam_diff/volume[i]
        
        dt  = min(dt, cfl * volume[i]/lam)
        
    return dt

def update_streamer_flux(ne_c:'float[:]', ni_c:'float[:]', rez_ne:'float[:]', rez_ni:'float[:]', 
                          dissip_ne:'float[:]', dissip_ni:'float[:]', src_ne:'float[:]', src_ni:'float[:]', 
                          dtime:'float', vol:'float[:]'):

    #$ omp parallel for 
    for i in range(len(ne_c)):
        ne_c[i]  += dtime  * ((rez_ne[i]  +  dissip_ne[i]) /vol[i] + src_ne[i] )
        ni_c[i]  += dtime  * ((rez_ni[i]  +  dissip_ni[i]) /vol[i] + src_ni[i] )
