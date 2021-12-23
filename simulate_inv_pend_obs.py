# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:27:39 2021

@author: ptrem
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

def simulate(x,t,L,M,dT, name):
    
    f = np.int(1/(dT*10))
    
    fig,ax = plt.subplots()
    p_pend, = plt.plot([],[],marker='o',\
                       linewidth=4,markersize=20,markerfacecolor='orange',\
                           color = 'k', markeredgecolor='k')
    p_cart, = plt.plot([],[],'ks',markersize=30, linewidth=4,\
                       markerfacecolor='orange',\
                           color='orange', markeredgecolor='k')
    p_obs, = plt.plot(0,2, marker='o', markersize=20, linewidth=4,\
                       color = 'k', markeredgecolor= 'k')
    plt.plot([-3,3],[-0.2,-0.2],'k-',lw=4)
    plt.plot([1,1],[-0.2,3],'k:',lw=2)
    plt.plot([-1,-1],[-0.2,3],'k:',lw=2)
    
    x_plot = x[::f,:]
    t_plot = t[::f]
    
    def init():
        ax.set_xlim(-3,3)
        ax.set_ylim(-0.5, 3)
        return x
    
    def animate(iter):
        x_iter = x_plot[iter,0]
        th_iter = x_plot[iter,2]
        
        p_cart.set_data(x_iter,0)
        p_pend.set_data(x_iter+np.array([0,L*np.sin(th_iter)]),\
                        0+np.array([0,-L*np.cos(th_iter)]))
        return p_pend
    
    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=len(t_plot),interval=50,blit=False,repeat=False)
    
    anim.save(name, fps=20, extra_args=['-vcodec', 'libx264'])
    return