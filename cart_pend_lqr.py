# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 08:12:48 2021

@author: ptrem
"""
import numpy as np
import control
import scipy
import matplotlib.pyplot as plt
import simulate_inv_pend as sim

def cartpend(x, t, m, M, L, g, d, uf):
    
    u = uf(x)
    
    dx = np.zeros((4,))

    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L**2*(M+m*(1-Cx**2))
    
    dx[0] = x[1]
    dx[1] = (1/D)*(-m**2*L**2*g*Cx*Sx+m*L**2*(m*L*x[3]**2*Sx-d*x[1]))+m*L**2*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx-m*L*Cx*(m*L*x[3]**2*Sx-d*x[1]))-m*L*Cx*(1/D)*u
    
    return dx

def cartpend_lin(x, t, m, M, L, g, d , uf):
    
    u = uf(x)
    
    A = np.array([[0, 1, 0, 0],
     [0, -d/M, m*g/M, 0],
     [0, 0, 0, 1],
     [0, -d/(M*L), -(m+M)*g/(M*L), 0]])
    
    B = np.reshape([0, 1/M, 0, 1/(M*L)], (-1,1))
    
    return np.squeeze(A@x.reshape(-1,1)+B*u)

m = 1
M = 5
L = 2
g = -9.81
d = 1

theta_start = 0.1

A = np.array([[0, 1, 0, 0],
     [0, -d/M, m*g/M, 0],
     [0, 0, 0, 1],
     [0, -d/(M*L), -(m+M)*g/(M*L), 0]])

B = np.reshape([0, 1/M, 0, 1/(M*L)], (4,1))
w,v = np.linalg.eig(A)

print('Eigenvalues of System Matrix: {}'.format(w))

ctrb_rank = np.linalg.matrix_rank(control.ctrb(A,B))

if ctrb_rank == np.shape(A)[0]:
    print('The rank of the control matrix is {} this means the system is controlable.'.format(ctrb_rank))
else:
    print('The rank of the control matrix is {} this means the system is not controlable.'.format(ctrb_rank))

Q = np.eye(4)

R = .01

K,S,E = control.lqr(A,B,Q,R)

T = 0.001

tspan = np.arange(0,10,T)
x0 = np.array([-1,0,np.pi+theta_start,0])
wr = np.array([1,0,np.pi,0])

u = lambda x: -K@(x-wr)

x = scipy.integrate.odeint(cartpend,x0,tspan,args=(m,M,L,g,d,u))

plt.figure()
plt.plot(tspan, x[:,0], label='x')
plt.plot(tspan, x[:,1], label='v')
plt.plot(tspan, x[:,2], label='$\\theta$')
plt.plot(tspan, x[:,3], label='$\omega$')
plt.grid()
plt.legend(loc='lower right', fontsize=16)
plt.title('LQR Control inverted Pendulum on a Cart', fontsize=20)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('States', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

sim.simulate(x,tspan,L,M,T,'Simulation_LQR.mp4')