# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys




def ECDF_representation_1d( D, n ):
    X = []
    D = np.array(D)
    x, f = ecdf(D + np.random.randn(len(D))*0.01)
#    ll = np.interp(f, x, np.linspace(0,1,n) )
    ll = np.interp(np.linspace(0,1,n), x, f)
    X = ll
    
    return X
#%%

       
#D=np.random.randn(100) 
#ecdf_D=ECDF_representation_1d(D,15)
#
#
#
#plt.plot(ecdf_D)