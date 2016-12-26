# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:03:59 2016
@author: David
"""
import scipy as sp
import scipy.optimize as so
import scipy.linalg as sl
import numpy as np
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
import matplotlib.pyplot as pl

mS = 3.0e-4 # Mass of sleeve [kg]
JS = 5.0e-9 # Moment of inertia of the sleeve [kgm]
mB = 4.5e-3 # Mass of bird [kg]
masstotal=mS+mB # total mass
JB = 7.0e-7 # Moment of inertia of bird [kgm]
r0 = 2.5e-3 # Radius of the bar [m]
rS = 3.1e-3 # Inner Radius of sleeve [m]
hS = 5.8e-3 # 1/2 height of sleeve [m]
lS = 1.0e-2 # verical distance sleeve origin to spring origin [m]
lG = 1.5e-2 # vertical distance spring origin to bird origin [m]
hB = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]
lB = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
cp = 5.6e-3 # rotational spring constant [N/rad]
g  = 9.81 #  [m/s^2]
global peck
peck = 0


def woodpecker(t,y,yp,sw):
    """
    input vectors of the form:
    y = [z, phiS, phiB, zp, phiSp, phiBp, lambda1, lambda2]
        [0,    1,    2,  3,      4,    5,       6,       7]
    yp = [zp, phiSp, phiBp, zpp, phiSpp, phiBpp, lambda1p, lambda2p]
         [ 0,     1,     2,   3,      4,      5,        6,        7]
    """    
    if sw[0]:
        res = np.zeros(np.size(y))
        res[0] = (mS +  mB) * yp[3] +  mB * lS * yp[4] + mB * lG * yp[5] + (mS + mB) * g
        res[1] = (mB * lS) * yp[3] + (JS + mB * lS * lS) * yp[4] + (mB * lS * lG) * yp[5] - cp * (y[2] - y[1]) + mB * lS * g + y[6]
        res[2] = mB * lG * yp[3] + (mB * lS * lG) * yp[4] + (JB +  mB * lG * lG) * yp[5] -  cp * (y[1] - y[2]) + mB * lG * g + y[7]
        res[3] = y[6]
        res[4] = y[7]
        res[5] = y[3] - yp[0]
        res[6] = y[4] - yp[1]
        res[7] = y[5] - yp[2]
        
    elif sw[1]:
        res = np.zeros(np.size(y))
        res[0] = (mS +  mB) * yp[3] +  mB * lS * yp[4] + mB * lG * yp[5] + (mS + mB) * g + y[7]
        res[1] = (mB * lS) * yp[3] + (JS + mB * lS * lS) * yp[4] + (mB * lS * lG) * yp[5] - cp * (y[2] - y[1]) + mB * lS * g + hS * y[6] + rS * y[7]
        res[2] = mB * lG * yp[3] + (mB * lS * lG) * yp[4] + (JB +  mB * lG * lG) * yp[5] -  cp * (y[1] - y[2]) + mB * lG * g
        res[3] = (rS -  r0) +  hS * y[1]
        res[4] = yp[0] +  rS * yp[1]
        res[5] = y[3] - yp[0]
        res[6] = y[4] - yp[1]
        res[7] = y[5] - yp[2]
        
    elif sw[2]:
        res = np.zeros(np.size(y))
        res[0] = (mS +  mB) * yp[3] +  mB * lS * yp[4] + mB * lG * yp[5] + (mS + mB) * g + y[7]
        res[1] = (mB * lS) * yp[3] + (JS + mB * lS * lS) * yp[4] + (mB * lS * lG) * yp[5] - cp * (y[2] - y[1]) + mB * lS * g - hS * y[6] + rS * y[7]
        res[2] = mB * lG * yp[3] + (mB * lS * lG) * yp[4] + (JB +  mB * lG * lG) * yp[5] -  cp * (y[1] - y[2]) + mB * lG * g
        res[3] = (rS -  r0) -  hS * y[1]
        res[4] = yp[0] +  rS * yp[1]
        res[5] = y[3] - yp[0]
        res[6] = y[4] - yp[1]
        res[7] = y[5] - yp[2]
        
    else:
        print('No time-steps should be taken in this state')
        return nan             
    
    return res

def state_events(t, y, yp, sw):
    # Had to be split up into two cases, otherwise assimulo is upset about y[6] always being zero    
    if sw[0]:   
        e = np.zeros((2,))
        # State I to II
        e[0] = hS * y[1] + ( rS -  r0)
        # State I to III
        e[1] = hS * y[1] - ( rS -  r0)

    elif sw[1]:
        e = np.zeros((2,))
        # State II to I
        e[0] = y[6]
        e[1] = 1

    else: 
        e = np.zeros((2,))
        # State III to I
        e[0] = y[6]  
        # State III to IV
        e[1] = hB * y[2] - (lS + lG - lB - r0)
    
    
    return e


def handle_event(solver, event_info):
    
    phiBp = solver.yd[2]
    state_info = event_info[0]
  
    # State I to II
    if (solver.sw[0] and phiBp < 0 and state_info[0]):
        print('State I')
        locked_sleeve(solver)
        solver.sw=[0,1,0,0]
    
    # State I to III
    elif (solver.sw[0] and phiBp > 0 and state_info[1]): 
        print('State I')
        locked_sleeve(solver)
        solver.sw=[0,0,1,0]
            
    # State II to I
    elif (solver.sw[1] and state_info[0]):
        print('State II')
        solver.sw=[1,0,0,0]
    
    # State III to I
    elif (solver.sw[2] and phiBp < 0 and state_info[0]):
        print('State III')
        solver.sw=[1,0,0,0]
            
    # State III to IV and back to III
    elif (solver.sw[2] and phiBp > 0 and state_info[1]):
        print('State IV')
        solver.y[5] = -solver.y[5]
        solver.yd[2] = -solver.yd[2]

        # Counting pecks
        global peck
        peck = peck + 1
            

def locked_sleeve(solver):
    """
    When the sleeve stops variables change in a discontinous fashion
    """
    y = solver.y
    yp = solver.yd
    
    phiBp = ((mB * lG) * yp[0] + (mB * lS * lG) * yp[1] + (JB + mB * lG * lG) * yp[2]) / (JB + mB * lG * lG)
    
    y[3] = 0
    y[4] = 0    
    y[5] = phiBp    
    
    yp[0] = 0
    yp[1] = 0    
    yp[2] = phiBp
    
    solver.y = y
    solver.yd = yp 


          
t0 = 0;
startsw = [1,0,0,0]

a = 1.5
y0 = np.array([5.0, 0.0, 0.0, 0.0, a, 2*a, 0.0, 0.0])
yd0 =  np.array([0.0, a, 2*a, -g, 0.0, 0.0, 0.0, 0.0])

problem = Implicit_Problem(woodpecker, y0, yd0, t0, sw0=startsw)

problem.state_events = state_events
problem.handle_event = handle_event
problem.name = 'Woodpecker'

sim = IDA(problem)
sim.rtol = 1e-6

#Tolerance of angular velocity
sim.atol[[4, 5]] = 1e10

#Tolerance of lambda
sim.atol[[6, 7]] = 1e10

ncp = 400

tfinal = 2.0
t, y, yd = sim.simulate(tfinal, ncp)

#sim.plot()
pl.plot(t, y[:,[0,]])

