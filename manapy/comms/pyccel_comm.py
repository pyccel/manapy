def define_halosend(w_c:'float[:]', w_halosend:'float[:]', indsend:'int[:]'):
    w_halosend[:] = w_c[indsend[:]]