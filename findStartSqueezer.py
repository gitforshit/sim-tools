# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:31:24 2016
@author: Axel
"""
from scipy import optimize as op
from  scipy import *
from  pylab import *

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

def constraints(q):
    q1 = 2
    q2,q3,q4,q5,q6,q7 = q
    res = zeros(6)
    
    xa,ya=-.06934,-.00227
    xb,yb=-0.03635,.03273
    xc,yc=.014,.072
    d,da,e,ea=28.e-3,115.e-4,2.e-2,1421.e-5
    rr,ra=7.e-3,92.e-5
    ss,sa,sb,sc,sd=35.e-3,1874.e-5,1043.e-5,18.e-3,2.e-2
    ta,tb=2308.e-5,916.e-5
    u,ua,ub=4.e-2,1228.e-5,449.e-5
    zf,zt=2.e-2,4.e-2
    fa=1421.e-5
    res[0] = rr * cos(q1) - d * cos(q1 + q2) - ss * sin(q3) - xb
    res[1] = rr * sin(q1) - d * sin(q1 + q2) + ss * cos(q3) - yb
    res[2] = rr * cos(q1) - d * cos(q1 + q2) - e * sin(q4 + q5) - zt * cos(q5) - xa
    res[3] = rr * sin(q1) - d * sin(q1 + q2) + e * cos(q4 + q5) - zt * sin(q5) - ya
    res[4] = rr * cos(q1) - d * cos(q1 + q2) - zf * cos(q6 + q7) - u * sin(q7) - xa
    res[5] = rr * sin(q1) - d * sin(q1 + q2) - zf * cos(q6 + q7) + u * cos(q7) - ya
    
#    g=zeros((6,))
#    g[0] = rr*cobe - d*cobeth - ss*siga - xb
#    g[1] = rr*sibe - d*sibeth + ss*coga - yb
#    g[2] = rr*cobe - d*cobeth - e*siphde - zt*code - xa
#    g[3] = rr*sibe - d*sibeth + e*cophde - zt*side - ya
#    g[4] = rr*cobe - d*cobeth - zf*coomep - u*siep - xa
#    g[5] = rr*sibe - d*sibeth - zf*siomep + u*coep - ya
    return res
    
y_1 = array([-0.0617138900142764496358948458001,  #  beta
    0.455279819163070380255912382449,   # gamma
    0.222668390165885884674473185609,   # phi
    0.487364979543842550225598953530,   # delta
    -0.222668390165885884674473185609,  # Omega
    1.23054744454982119249735015568])   #epsilon


    
guess = zeros(6) + 0.4


ans = op.fsolve(constraints, guess)
print(ans)
print(constraints(ans))
#Gprime = getGradient(constraints)
#print(Gprime(y_1))
##print(str(constraints(guess)))
##print(str(Gprime(guess)))
#for i in range(1000):
#    #print(i)
#    Dx = solve(Gprime(guess), -constraints(guess))
#    guess = (guess + Dx) % (2 * pi)
#    su = sum(abs(constraints(guess)))
#    if su < tol:
#        print(str(su)+str(guess))
#        break
#print('end')





