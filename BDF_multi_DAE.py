"""
Created on Thu Nov 10 20:21:45 2016
@author: David
"""
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
    
def evaluateGradient(function,x,epsilon = 1e-8):
    """Evaluates the gradient of function in point x using finite difference 
        approximation.
        
        Args:
            function: function object to find the gradient of.
            x: nparray object contining point to evaluate in.
            epsilon: {Optional} {Default = 1e-5} Finite difference step size.
            
        Returns:
            nparray object containing the value of the gradient in point x.
    """
    h = np.zeros(np.shape(x))
    # Assumes that the input and the output has the same length. Needs generalization
    res = np.zeros((np.shape(x)[0],np.shape(x)[0]))
    for i in range(0,len(x)):
        # Set the step on the correct variable.
        h[i] = epsilon
        # Approximate derivative using central difference approximation.
        res[i] = (function(x + h) - function(x - h)) / (2 * epsilon)
        # Reset step for next iteration.
        h[i] = 0.0
    return res


class BDF(Explicit_ODE):
    """
    Superclass to Explicit Euler (BDE_1), BDE_3, BDE_4
    """
    tol = 1.e-6
    maxit = 100
    maxsteps = 20000
    order_method = 1
    solve_as_DAE = True # If true solves for F(t,x,x')=0, otherwise for f(t,x)=x'
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.01 #0.01
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
        
        self.error = []
        self.hrdct = 0
    
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
        order_method = self.order_method
        
        #Lists for storing the result
        tres = []
        yres = []
        tres.append(t)
        yres.append(y)
        reduced = False
        t_np1 = t
        
        for i in range(self.maxsteps):
            if t_np1 >= tf:
                break
            self.statistics["nsteps"] += 1
            
            if i<order_method:  # initial steps
                t_np1, y_np1 = self.step(list(reversed(tres)), list(reversed(yres)), h)
            else:   
                try:
                    t_np1, y_np1 = self.step(list(reversed(tres[-order_method:])), list(reversed(yres[-order_method:])), h)
                except (Explicit_ODE_Exception, SL.LinAlgError) as E:
                    self._set_h(h/2)
                    self.hrdct += 1
                    _, tred, yred = self.integrate(t, y, tf, opts)
                    reduced = True
                    print('h reduced')
                        
            
            if reduced:
                tres = tres + tred
                yres = yres + yred
                t = tred[-1]
                y = yred[-1]
            else:
                tres.append(t_np1)
                yres.append(y_np1.copy())
        
            h=min(self.h,np.abs(tf-t))
        else:
#            print(t)
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def step(self, T, Y, h):
        """
        BDF with order=len(Y) with Newton Iteration and Zero order predictor
        """
        order = len(Y)
        if order == 1:
            alpha = [1., -1.]
        elif order == 2:
            alpha=[3./2., -2., 1./2]
        elif order == 3:
            alpha=[11./6., -3., 3./2., -1./3.]
        elif order == 4:
            alpha=[25./12., -4., 3., -4./3., 1./4.]
        else:
            print('Methods of higher order than 4 are not implemented')
            exit()
        
        F=self.problem.rhs
        
        t_np1 = T[0] + h
        y_np1_i = Y[0]

        old_y_alpha_sum = 0
        for i in range(order):
            old_y_alpha_sum = old_y_alpha_sum + Y[i] * alpha[i + 1]
        
        if self.solve_as_DAE:
            def Hfunc():
                def H(y):
                    return F(t_np1, y, (old_y_alpha_sum + y * alpha[0]) / h)
                return H
            H = Hfunc()    
            Hprime = getGradient(H)
        else:
            def Gfunc():
                def G(y):
                    return alpha[0] * y + old_y_alpha_sum - h * F(t_np1,y)
                return G
            H = Gfunc()#G=Gfunc()    
            Hprime = getGradient(H)#Gprime=getGradient(G)
               
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            Dx=SL.solve(Hprime(y_np1_i),-H(y_np1_i))
            y_np1_ip1=Dx+y_np1_i            
            
            normY = SL.norm(y_np1_ip1-y_np1_i)
            if normY < self.tol:
                self.error.append(normY)
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
        
        
    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        return t + h, y + h*f(t, y) 
        

    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of step reductions      : {reduction} '.format(reduction=self.hrdct), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)

        self.log_message('\nSolver options:\n',verbose)
        self.log_message(' Solver            : BDF2',verbose)
        self.log_message(' Solver type       : Fixed step',verbose)
        self.log_message(' Corrector type    : Newton\n',verbose)


def spring_pend_f(t,y):
    ydot = np.zeros(4)
    ydot[0] = y[2]
    ydot[1] = y[3]
    ydot[2] = -y[0] * lambda_fkt(y[0], y[1])
    ydot[3] = -y[1] * lambda_fkt(y[0], y[1]) - 1
    return ydot

def spring_pend_F(t,y,ydot):
    res = np.zeros(4)
    res[0] = ydot[0] - y[2]
    res[1] = ydot[1] - y[3]
    res[2] = ydot[2] + y[0] * lambda_fkt(y[0], y[1])
    res[3] = ydot[3] + y[1] * lambda_fkt(y[0], y[1]) + 1
    return res
        
def lambda_fkt(y1, y2):
    k = 15
    return k * (np.sqrt(y1**2 + y2**2) - 1) / np.sqrt(y1**2 + y2**2)
        
y0 = np.array([1.05, 0., 0., 0.])
pend_mod=Explicit_Problem(spring_pend_F, y0)
pend_mod.name='Spring Pendulum'
  

#Define an explicit solver
exp_sim_2 = BDF(pend_mod) #Create a BDF solver

exp_sim_CV = CVode(pend_mod)
exp_sim_CV.atol=1.e-8
exp_sim_CV.rtol=1.e-8

exp_sim_2.order_method = 4
#exp_sim_2.solve_as_DAE = False

t, y = exp_sim_2.simulate(5)
#to_plot = round(size(t)/10)
#t = t[-to_plot:]
#y = y[-to_plot:]
#im = plot(t,y)

#labels = ('x', 'y', 'vx', 'vy')
#plt.legend(im, labels)

#t, y = exp_sim_CV.simulate(100)
#to_plot = round(size(t)/10)
#t = t[-to_plot:]
#y = y[-to_plot:]
#im = plot(t,y,'--')


plt.xlabel('time')

exp_sim_2.plot()
mpl.show()