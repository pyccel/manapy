from pyccel.decorators import stack_array

def initialisation_gaussian_2d(ne:'float[:]',  u:'float[:]', v:'float[:]', w:'float[:]', 
                               P:'float[:]', center:'float[:,:]', Pinit:'float'):
    from numpy import exp    
    nbelements = len(center)
    
    sigma = 0.05
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        
        ne[i] = 5 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2) / sigma**2) + 1
        u[i]  = 0.
        v[i]  = 0.
        w[i]  = 0.
        P[i]  = Pinit * (.5 - xcent)

def initialisation_gaussian_3d(ne:'float[:]',  u:'float[:]', v:'float[:]', w:'float[:]', 
                               P:'float[:]', center:'float[:,:]', Pinit:'float'):
    from numpy import exp    
    nbelements = len(center)
    
    sigma = 0.05
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        zcent = center[i][2]
        
        ne[i] = 5 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2 +  (zcent-0.45)**2) / sigma**2) + 1
        u[i]  = 0.
        v[i]  = 0.
        w[i]  = 0.
        P[i]  = Pinit * (.5 - xcent)
  
@stack_array('norm') 
def time_step(u:'float[:]', v:'float[:]', w:'float[:]',  cfl:'float', normal:'float[:,:]', 
              mesure:'float[:]', volume:'float[:]', faceid:'int[:,:]', dim:'int',
              Dxx:'float', Dyy:'float', Dzz:'float'):
   
    from numpy import sqrt, fabs, zeros
    
    nbelement =  len(faceid)
    u_n = 0.
    norm = zeros(3)
    dt = 1e6
  
    for i in range(nbelement):
        lam = 0.
       
        for j in range(dim+1):
            norm[:] = normal[faceid[i][j]][:]
            
            u_n = fabs(u[i]*norm[0] + v[i]*norm[1] + w[i]*norm[2])
            lam_convect = u_n/mesure[faceid[i][j]] 
            lam += lam_convect * mesure[faceid[i][j]]
        
            mes = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2])
            lam_diff = Dxx * mes**2 + Dyy * mes**2 + Dzz * mes**2
            lam += lam_diff/volume[i]
        
        dt  = min(dt, cfl * volume[i]/lam)
     
    return dt


def update_new_value(ne_c:'float[:]', u_c:'float[:]', v_c:'float[:]', P_c:'float[:]', 
                     rez_ne:'float[:]', dissip_ne:'float[:]',  src_ne:'float[:]',
                     dtime:'float', vol:'float[:]'):

    ne_c[:]  += dtime  * ((rez_ne[:]  +  dissip_ne[:]) /vol[:] + src_ne[:] )
