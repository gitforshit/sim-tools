from scipy import *
from assimulo.problem import Implicit_Problem #Imports the problem formulation from Assimulo
from assimulo.solvers import IDA #Imports the solver IDA from Assimulo

def init_squeezer():
	y_1 = array([-0.0617138900142764496358948458001,  #  beta
				0.,                                 #  theta
				0.455279819163070380255912382449,   # gamma
				0.222668390165885884674473185609,   # phi
				0.487364979543842550225598953530,   # delta
				-0.222668390165885884674473185609,  # Omega
				1.23054744454982119249735015568])   #epsilon
	lamb = array([
			98.5668703962410896057654982170,        # lambda[0]
			-6.12268834425566265503114393122])       # lambda[1]			
	y=hstack((y_1,zeros((7,)),lamb,zeros((4,))))
	yp=hstack((zeros(7,),array([
			14222.4439199541138705911625887,        #  betadotdot
			-10666.8329399655854029433719415,       #  Thetadotdot
			0.,0.,0.,0.,0.]),zeros((6,))))
	return y,yp
def squeezer1 (t, y, yp):
    """
    Residual function of the 7-bar mechanism in
    Hairer, Vol. II, p. 533 ff, see also formula (7.11)
    written in residual form
    y,yp vector of dim 20, t scalar
    """
    # Inertia data
    m1,m2,m3,m4,m5,m6,m7=.04325,.00365,.02373,.00706,.07050,.00706,.05498
    i1,i2,i3,i4,i5,i6,i7=2.194e-6,4.410e-7,5.255e-6,5.667e-7,1.169e-5,5.667e-7,1.912e-5
    # Geometry
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
    # Driving torque
    mom=0.033
    # Spring data
    c0=4530.
    lo=0.07785
    # Initial computations and assignments
    beta,theta,gamma,phi,delta,omega,epsilon=y[0:7]
    bep,thp,gap,php,dep,omp,epp=y[7:14]
    lamb=y[14:20]
    sibe,sith,siga,siph,side,siom,siep=sin(y[0:7])
    cobe,coth,coga,coph,code,coom,coep=cos(y[0:7])
    
    sibeth = sin(beta+theta);cobeth = cos(beta+theta)
    siphde = sin(phi+delta);cophde = cos(phi+delta)
    siomep = sin(omega+epsilon);coomep = cos(omega+epsilon)


    # Mass matrix
    m=zeros((7,7))
    m[0,0] = m1*ra**2 + m2*(rr**2-2*da*rr*coth+da**2) + i1 + i2
    m[1,0] = m[0,1] = m2*(da**2-da*rr*coth) + i2
    m[1,1] = m2*da**2 + i2
    m[2,2] = m3*(sa**2+sb**2) + i3
    m[3,3] = m4*(e-ea)**2 + i4
    m[4,3] = m[3,4] = m4*((e-ea)**2+zt*(e-ea)*siph) + i4
    m[4,4] = m4*(zt**2+2*zt*(e-ea)*siph+(e-ea)**2) + m5*(ta**2+tb**2)+ i4 + i5
    m[5,5] = m6*(zf-fa)**2 + i6
    m[6,5] = m[5,6] = m6*((zf-fa)**2-u*(zf-fa)*siom) + i6
    m[6,6] = m6*((zf-fa)**2-2*u*(zf-fa)*siom+u**2) + m7*(ua**2+ub**2)+ i6 + i7
    
    #   Applied forces
    
    xd = sd*coga + sc*siga + xb
    yd = sd*siga - sc*coga + yb
    lang  = sqrt ((xd-xc)**2 + (yd-yc)**2)
    force = - c0 * (lang - lo)/lang
    fx = force * (xd-xc)
    fy = force * (yd-yc)
    ff=array([mom - m2*da*rr*thp*(thp+2*bep)*sith,
        m2*da*rr*bep**2*sith,
        fx*(sc*coga - sd*siga) + fy*(sd*coga + sc*siga),
        m4*zt*(e-ea)*dep**2*coph,
        - m4*zt*(e-ea)*php*(php+2*dep)*coph,
        - m6*u*(zf-fa)*epp**2*coom,
        m6*u*(zf-fa)*omp*(omp+2*epp)*coom])
      
    #  constraint matrix  G
      
    gp=zeros((6,7))
	
    gp[0,0] = - rr*sibe + d*sibeth
    gp[0,1] = d*sibeth
    gp[0,2] = - ss*coga
    gp[1,0] = rr*cobe - d*cobeth
    gp[1,1] = - d*cobeth
    gp[1,2] = - ss*siga
    gp[2,0] = - rr*sibe + d*sibeth
    gp[2,1] = d*sibeth
    gp[2,3] = - e*cophde
    gp[2,4] = - e*cophde + zt*side
    gp[3,0] = rr*cobe - d*cobeth
    gp[3,1] = - d*cobeth
    gp[3,3] = - e*siphde
    gp[3,4] = - e*siphde - zt*code
    gp[4,0] = - rr*sibe + d*sibeth
    gp[4,1] = d*sibeth
    gp[4,5] = zf*siomep
    gp[4,6] = zf*siomep - u*coep
    gp[5,0] = rr*cobe - d*cobeth
    gp[5,1] = - d*cobeth
    gp[5,5] = - zf*coomep
    gp[5,6] = - zf*coomep - u*siep
    #     Index-1 constraint
    gqq = zeros((6,))
    gqq[0] = - rr * cobe * yp[0]**2 + d * cobeth * (yp[0] + yp[1])**2 + ss * siga * yp[2]**2
    gqq[1] = - rr * sibe * yp[0]**2 + d * sibeth * (yp[0] + yp[1])**2 - ss * coga * yp[2]**2
    gqq[2] = - rr * cobe * yp[0]**2 + d * cobeth * (yp[0] + yp[1])**2 + e * siphde * (yp[3] + yp[4])**2 + zf * code * yp[4]**2
    gqq[3] = - rr * sibe * yp[0]**2 + d * sibeth * (yp[0] + yp[1])**2 - e * cophde * (yp[3] + yp[4])**2 + zf * side * yp[4]**2
    gqq[4] = - rr * cobe * yp[0]**2 + d * cobeth * (yp[0] + yp[1])**2 + zf * coomep * (yp[5] + yp[6])**2 + u * siep * yp[6]**2
    gqq[5] = - rr * sibe * yp[0]**2 + d * sibeth * (yp[0] + yp[1])**2 + zf * siomep * (yp[5] + yp[6])**2 - u * coep * yp[6]**2

    #     Construction of the residual

    res_1 = yp[0:7] - y[7:14]
    res_2 = dot(m,yp[7:14])- ff[0:7]+dot(gp.T,lamb)
    res_3 = dot(gp,y[7:14])
	
    return hstack((res_1,res_2,res_3))
 
y0, yd0 = init_squeezer()
t0 = 0
model = Implicit_Problem(squeezer1, y0, yd0, t0) #Create an Assimulo problem
sim = IDA(model)
ATOL = hstack((ones(7)*1e-6, ones(13)*1e5))
sim.atol = ATOL
t, y, ud = sim.simulate(0.03)

y7 = (y[:,0:7] + pi) % (2 * pi) - pi
y_lambda = (y[:,14:20])# + pi) % (2 * pi) - pi
plot(t, y_lambda)
#sim.plot()

	