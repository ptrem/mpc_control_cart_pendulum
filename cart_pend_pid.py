# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 08:12:48 2021

@author: ptrem
"""
import numpy as np
from time import time
import scipy.integrate
import matplotlib.pyplot as plt
import simulate_inv_pend as sim

def cartpend(x, t, m, M, L, g, d, u):
    
    dx = np.zeros((4,))

    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L**2*(M+m*(1-Cx**2))
    
    dx[0] = x[1]
    dx[1]= (1/D)*(-m**2*L**2*g*Cx*Sx+m*L**2*(m*L*x[3]**2*Sx-d*x[1]))+m*L**2*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx-m*L*Cx*(m*L*x[3]**2*Sx-d*x[1]))-m*L*Cx*(1/D)*u
    
    return dx

def pid_controler(error, previous_error, integral, time_delta):
    
    # The gains were emicically tuned
    Kp = -160
    Kd = -40
    Ki = -40
    
    derivative = (error - previous_error)/time_delta
    integral += error * time_delta
    F = (Kp * error) + (Kd * derivative) + (Ki * integral)
    return F, integral

def get_error(x, x_ref):
    #state_error = x - x_ref
    state_error = (x[2] % (2 * np.pi)) - x_ref[2]
    
    if state_error > np.pi:
        state_error = state_error - (2 * np.pi)
    return state_error

m = 1
M = 5
L = 2
g = -9.81
d = 1

sim_time = 10                # final time for simulation
nsteps = 1000              # number of time steps
time_delta = sim_time/(nsteps-1)   # how long is each time step?
ts = np.linspace(0,sim_time,nsteps) # linearly spaced time vector

time_sup = 60                   # suppression time
previous_timestamp = 0

diff_theta0 = -0.2

x_init = -1
dx_init = 0
theta_init = np.pi+diff_theta0
dtheta_init = 0

x_ref = 1
dx_ref = 0
theta_ref = np.pi
dtheta_ref = 0

F = 0

x0 = np.array([x_init, dx_init, theta_init, dtheta_init])

current_state = x0
state_ref = np.array([x_ref, dx_ref, theta_ref, dtheta_ref])

integral = 0
previous_error = get_error(current_state, state_ref)

previous_time_delta = 0
current_timestamp = 0

cat_states = x0[:,np.newaxis]
cat_controls = []
i = 0

if __name__ == '__main__':
    while (current_timestamp < sim_time):
        current_timestamp = i*time_delta
        error = get_error(current_state, state_ref)
        

        F, integral = pid_controler(error, previous_error, integral, time_delta)
        
        next_state = scipy.integrate.odeint(cartpend,current_state,[0, time_delta],args=(m,M,L,g,d,F))
        next_state = next_state[-1,:]
        
        previous_time_delta = time_delta
        previous_timestamp = current_timestamp
        previous_error = error
        current_state = next_state
        
        cat_states = np.concatenate((cat_states, current_state[:,np.newaxis]), axis=1)
        cat_controls.append(F)
        
        print('Iteration: {}'.format(i))
        print('State: {}'.format(next_state))
        print('Control: {}'.format(F))
        
        i += 1
        
cat_controls = np.array(cat_controls)

plt.figure()
plt.plot(ts, cat_states[0,1:], label='x')
plt.plot(ts, cat_states[1,1:], label='v')
plt.plot(ts, cat_states[2,1:], label='$\\theta$')
plt.plot(ts, cat_states[3,1:], label='$\omega$')
plt.grid()
plt.legend(loc='lower right', fontsize=16)
plt.title('PID Control inverted Pendulum on a Cart', fontsize=20)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('States', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

x = np.moveaxis(cat_states,0,1)
sim.simulate(x,ts,L,M,time_delta,'Simulation_PID.mp4')