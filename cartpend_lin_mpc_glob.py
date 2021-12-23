# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:11:56 2021

@author: ptrem
"""
import casadi as ca
import numpy as np
from time import time
import simulate_inv_pend as sim
import matplotlib.pyplot as plt

m = 1
M = 5
L = 2
g = -9.81
d = 1 

theta0 = 0.2

N = 100
T = 0.02
sim_time = 10

x_max = 3
x_min = -x_max
theta_max = ca.inf
theta_min = -theta_max
F_max = 30 #N
F_min = -F_max

x_init = -1
v_init = 0
theta_init = ca.pi+theta0
omega_init = 0

x_ref = 1
v_ref = 0
theta_ref = ca.pi
omega_ref = 0

x_fp = -1
v_fp = 0
theta_fp = ca.pi
omega_fp = 0

t0 = 0
state_init = ca.DM([x_init, v_init, theta_init, omega_init])    # initial state
state_ref = ca.DM([x_ref, v_ref, theta_ref, omega_ref])      # target state
state_fp = ca.DM([x_fp, v_fp, theta_fp, omega_fp])

control_init = 0
control_fp = 0

t = ca.DM(t0)

Q_x = 0.5
Q_v = 1e-2
Q_theta = 1
Q_omega = 0
R = 1e-3

Q = ca.diagcat(Q_x, Q_v, Q_theta, Q_omega)

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())

x = ca.SX.sym('x')
v = ca.SX.sym('v')
theta = ca.SX.sym('theta')
omega = ca.SX.sym('omega')
states = ca.vertcat(
    x,
    v,
    theta,
    omega
)
n_states = states.numel()

# control symbolic variables
u = ca.SX.sym('u')
controls = ca.vertcat(
    u
)

xP = ca.SX.sym('xP')
vP = ca.SX.sym('vP')
thetaP = ca.SX.sym('thetaP')
omegaP = ca.SX.sym('omegaP')
XP = ca.vertcat(
    xP,
    vP,
    thetaP,
    omegaP
)

uP = ca.SX.sym('uP')
UP = ca.vertcat(
    uP
)

n_controls = controls.numel()

X = ca.SX.sym('X', n_states, N + 1)
U = ca.SX.sym('U', n_controls, N)
P = ca.SX.sym('P', 2*n_states)

def cartpend(x, u):
    
    m = 1
    M = 5
    L = 2
    g = -9.81
    d = 1
    
    Sx = ca.sin(x[2])
    Cx = ca.cos(x[2])
    D = m*L**2*(M+m*(1-Cx**2))
    
    dxdt = [x[1],
            (1/D)*(-m**2*L**2*g*Cx*Sx+m*L**2*(m*L*x[3]**2*Sx-d*x[1]))+m*L**2*(1/D)*u,
            x[3],
            (1/D)*((m+M)*m*g*L*Sx-m*L*Cx*(m*L*x[3]**2*Sx-d*x[1]))-m*L*Cx*(1/D)*u
            ]
    return ca.vertcat(*dxdt)


def cartpend_lin(x, u, xp, up):
    
    m = 1
    M = 5
    L = 2
    g = -9.81
    d = 1
    
    A = ca.DM([[0, 1, 0, 0],
         [0, -d/M, m*g/M, 0],
         [0, 0, 0, 1],
         [0, -d/(M*L), -(m+M)*g/(M*L), 0]])
    
    B = ca.DM(np.array([0, 1/M, 0, 1/(M*L)]))
    
    return cartpend(xp,up)+A@(x-xp)+B@(u-up)

cost_fn = 0  # cost function
x0 = P[:n_states]  # constraints in the equation
xref = P[-n_states:]

RHS_lin = cartpend_lin(states, controls, XP, UP)
f_lin = ca.Function('f', [states, controls, XP, UP], [RHS_lin])

RHS = cartpend(states, controls)
f =  ca.Function('f', [states, controls], [RHS])

g = X[:,0]-x0

for k in range(N):
    st = X[:,k]
    con = U[:,k]
    st_next = X[:,k+1]
    
    cost_fn = cost_fn+((st-xref).T@Q@(st-xref))+con.T@R@con
    
    f_value = f_lin(st, con, state_fp, control_init)
    st_next_euler = st+(T*f_value)
    g = ca.vertcat(g, st_next-st_next_euler)

#for k in range(1,N):
#    cost_fn = cost_fn + (U[:,k]-U[:,k-1]).T@R@(U[:,k]-U[:,k-1])
    
OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1)+n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1)+n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = x_min     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = theta_min     # theta lower bound
lbx[3: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = x_max      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = theta_max      # theta upper bound
ubx[3: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbx[n_states*(N+1):] = F_min                  # v lower bound for all V
ubx[n_states*(N+1):] = F_max                  # v upper bound for all V

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}


u0 = ca.DM.zeros((n_controls, N))          # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full

cost_cat = []

mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0)
times = np.array([[0]])

#control_init = 0

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (mpc_iter < sim_time/T):
        print('Iteration: {}'.format(mpc_iter))
        print('Linearization Point Xp: {}, Up: {}'.format(state_fp, control_fp))
        
        t1 = time()
        args['p'] = ca.vertcat(
            state_init, # current state
            state_ref   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states*(N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][:n_states*(N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u)
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(T, t0, state_init, u, f)

        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        times = np.vstack((
            times,
            t2-t1
            ))
        
        control_init = u0[:,0]
        
        #print('Next Point Xp: {}, Up: {}'.format(state_init, control_init))
        
        ss_error = ca.norm_2(state_init - state_ref)
        #print('Cost: {}'.format(ss_error))
        cost_cat.append(ss_error)
        
        mpc_iter = mpc_iter + 1
        
    tspan = np.arange(0,sim_time,T)
    
    print('avg iteration time: ', np.array(times).mean()*1000, 'ms')
        
    plt.figure()
    plt.plot(tspan, cat_states[0,0,:-1], label='x')
    plt.plot(tspan, cat_states[1,0,:-1], label='v')
    plt.plot(tspan, cat_states[2,0,:-1], label='$\\theta$')
    plt.plot(tspan, cat_states[3,0,:-1], label='$\omega$')
    plt.grid()
    plt.legend(loc='lower right', fontsize=16)
    plt.title('MPC Control inverted Pendulum on a Cart', fontsize=20)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('States', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_ref)
    
    tspan = np.arange(0,sim_time,T)
    x_sim = np.moveaxis(cat_states[:,0,:],0,-1)
    sim.simulate(x_sim,tspan,L,M,T,'Simulation_mpc_lin_glob.mp4')