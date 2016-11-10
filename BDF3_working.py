# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:19:32 2016

@author: poff
"""

from  scipy import *
from  pylab import *
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode


    

            

def getGradient(function):
    """Returns the gradient of a funciton as a function object. Uses finite 
        differences approximation.
    
        Args:
            function: function object to find the gradient of.
            
        Returns:
            function object of the gradient.
    """
    def grad(x):
       return evaluateGradient(function,x)         
    return grad
    
def evaluateGradient(function,x,epsilon = 1e-2):
    """Evaluates the gradient of function in point x using finite difference 
        approximation.
        
        Args:
            function: function object to find the gradient of.
            x: nparray object contining point to evaluate in.
            epsilon: {Optional} {Default = 1e-5} Finite difference step size.
            
        Returns:
            nparray object containing the value of the gradient in point x.
    """
    h = zeros(shape(x))
    res = zeros((shape(x)[0],shape(x)[0]))
    for i in range(0,len(x)):
        # Set the step on the correct variable.
        h[i] = epsilon
        # Approximate derivative using central difference approximation.
        res[i] = (function(x + h) - function(x - h)) / (2 * epsilon)
        # Reset step for next iteration.
        h[i] = 0.0
    return res


class BDF_2(Explicit_ODE):
    """
    BDF-2   (Example of how to set-up own integrators for Assimulo)
    """
    tol=1.e-6  
    maxit=100     
    maxsteps=5000
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.01
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]
        
    h=property(_get_h,_set_h)
        
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = []
        yres = []
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            
            if i==0:  # initial steps
                t_np1,y_np1 = self.step_EE(t,y, h)
            elif i==1:
                t_np1,y_np1 = self.step_EE(t,y, h)
                t_nm2 = t_nm1
                y_nm2 = y_nm1
            else:   
                t_np1, y_np1 = self.step_BDF3([t,t_nm1,t_nm2], [y,y_nm1,y_nm2], h)
                t_nm2 = t_nm1
                y_nm2 = y_nm1
            t,t_nm1=t_np1,t
            y,y_nm1=y_np1,y
            
            tres.append(t)
            yres.append(y.copy())
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        return t + h, y + h*f(t, y) 
        
    def step_BDF2(self,T,Y, h):
        """
        BDF-2 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[3./2.,-2.,1./2]
        f=self.problem.rhs
        
        t_n,t_nm1=T
        y_n,y_nm1=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1)+h*f(t_np1,y_np1_i))/alpha[0]
            if SL.norm(y_np1_ip1-y_np1_i) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)

    
    def step_BDF3(self,T,Y, h):
        """
        BDF-3 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1+alpha_3*y_nm2=h f(t_np1,y_np1)
        alpha=[11/6, -3, 3/2, -1/3]
        """
        alpha=[11./6., -3., 3./2., -1./3.]
        f=self.problem.rhs
        
        t_n,t_nm1,t_nm2=T
        y_n,y_nm1,y_nm2=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        
        def ftime(func,t):
            def ft(x):
                return func(x,t)
            return ft
            
        def Gfunc():
            def G(y):
                return alpha[0]*y+alpha[1]*y_n+alpha[2]*y_nm1+alpha[3]*y_nm2-h*f(t_np1,y)
            return G
        
        
        G=Gfunc()    
        Gprime=getGradient(G)

        
        
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1

            Dx=solve(evaluateGradient(G,y_np1_i),-G(y_np1_i))

            y_np1_ip1=Dx+y_np1_i
            
           # y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1+alpha[3]*y_nm2)+h*f(t_np1,y_np1_i))/alpha[0]
                    
            
            if SL.norm(y_np1_ip1-y_np1_i) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF2',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
            
#Define the rhs
def f(t,y):
    k = 2
    ydot = np.zeros(4)
    ydot[0] = y[2]
    ydot[1] = y[3]
    ydot[2] = -y[0] * lambda_fkt(y[0], y[1])
    ydot[3] = -y[1] * lambda_fkt(y[0], y[1]) - 1
    return ydot
#    ydot = -y[0]
#    return np.array([ydot])
    
def lambda_fkt(y1, y2):
    k = 100
    return k * (np.sqrt(y1**2 + y2**2) - 1) / np.sqrt(y1**2 + y2**2)
    
    
#Define an Assimulo problem
exp_mod = Explicit_Problem(f, 4)
exp_mod.name = 'Simple BDF-2 Example'

#Define another Assimulo problem
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return np.array([y[1],-gl*np.sin(y[0])])
    
#pend_mod=Explicit_Problem(pend, y0=np.array([2.*np.pi,1.]))
pend_mod=Explicit_Problem(f, y0=np.array([1., 0., 0., 0.]))
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = BDF_2(pend_mod) #Create a BDF solver
#exp_sim = CVode(pend_mod)
t, y = exp_sim.simulate(10)
exp_sim.plot()
mpl.show()






