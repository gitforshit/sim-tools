# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:41:40 2016
@author: David
"""
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode
import EEsolver as BDF

def spring_pend(t,y):
    ydot = np.zeros(4)
    ydot[0] = y[2]
    ydot[1] = y[3]
    ydot[2] = -y[0] * lambda_fkt(y[0], y[1])
    ydot[3] = -y[1] * lambda_fkt(y[0], y[1]) - 1
    return ydot
        
def lambda_fkt(y1, y2):
#    k = 200
    return k * (np.sqrt(y1**2 + y2**2) - 1) / np.sqrt(y1**2 + y2**2)


#n = array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150.])
#times = []
#for k in n:
#    
#    y0 = np.array([1.05, 0., 0., 0.])
#    pend_mod=Explicit_Problem(spring_pend, y0)
#    pend_mod.name='Spring Pendulum'    
#    
#    exp_sim = BDF.EE(pend_mod)    
#    
#    start = time.time()
#    t, y = exp_sim.simulate(10)
#    end = time.time()
#    times.append(end - start)
#
#times = array(times)
#plot(n,times)

k = 150
tfinal = 50

exp_sim = CVode(pend_mod)
t, y = exp_sim.simulate(tfinal)
mpl.plot(t,y,'--')

exp_sim = BDF.EE(pend_mod)
t, y = exp_sim.simulate(tfinal)
exp_sim.plot()

mpl.show()

