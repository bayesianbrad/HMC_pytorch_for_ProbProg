#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:44:23 2017

@author: bradley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:20:07 2017

@author: bradley
"""
import random
import math
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss


#def leap_frog_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
#            # Assumes mass is constant for all particles
#    p_prev = init_p
#    q_prev = init_q
#    
#    for i in range(steps):
#        # Plot phase space
#        plt.figure(3)
#        plt.plot(q_prev,p_prev,'r*')
#        
#        # leapfrog step added
#        p_leapfrog = p_prev - (step_size/2)*q_prev
#        q_next     = q_prev + (step_size*p_leapfrog)/m
#        p_next     = p_leapfrog - (step_size/2)*q_next
#        
#        # Update p and q                   
#        p_prev = p_next
#        q_prev = q_next 
#        print(q_prev, p_prev)
#        plt.hold(True)     

def grad_U(q ,sigma_inv):
    '''Takes a postion and a defined potential and returns the derivative
        assumes has dimensions [2 x 1]. '''
    return np.dot(sigma_inv,q)
    # Returns gradient of potential evaluated at q
    # Things to note:
        # q can be a vector.
    

def Upotential(q, sigma_inv):
    '''The potential is defined as U(q)  = -log(π(q))
    Example of bivariate normal with variances of 1 and 
    correlations 0.95'''
    # the potential function
    # TO COMPLETE
    # Returns the value of the potential evaluated at q
    # Things to note:
    return np.dot(np.dot(np.transpose(q), sigma_inv),q)

def Kkinetic(p):
    """ The kinetic energy is choosen to be p.p / 2 , with m = 1. """
    return  np.sum(np.dot(p,np.transpose(p)))/2 

def grad_K(p):
    ''' The gradient of the kinetic energy. Takes vector input q, composed
        of the componets of the "postions" our varibles of interest  '''
    return p

def ham(q,p,sigma_inv):
    '''Calcualtes the hamiltonain evaluated at the current q and p'''
    return Kkinetic(p) + Upotential(q,sigma_inv)

def hamiltonian_monte_carlo(step_size, steps, q_current, iternum):
    ''' Performs one interations of the HMC, and perfroms a trajectory with
        L leapfrog steps. The step returns the value of the Hamiltonian and 
        the new proposed q value (may be the same if rejected).  The p's are
        generated only within the function and then discarded.
        
        The canonical distribution 'the joint distribution' π(q,p) = π(q)π(p)
        The Hamiltonian is given by H(q,p) = U(q) + K(p) 
        Potential energy  U(q) = -log(π(q)) 
        Kinetic energy    K(p) = -log(π(p))
        
        '''
    # For the bivariate normal case we define sigma to be:
    sigma   = np.array([[1, 0.8],[0.8,1]])
    sig_inv = np.linalg.inv(sigma)
    
    
#==============================================================================
#     Iteration for leapfrog begins
#==============================================================================
    q       = q_current
    # draws from a normal N(0,1)
    if (iternum == 0):
        p = np.array([[-1], [1]])
    else:
        p = np.random.randn(np.shape(q)[0],1)  
    
    p_current = p
    
    # make half step for momentum at the beginning
    p = p - (step_size/2)*grad_U(q,sig_inv)
    
    # Alternate fulls teps for position and momentum
    
    for i in range(steps):
        # make half step for postion
        q = q + (step_size) * grad_K(p)
        # make full step for postion if i < steps
        
        if (i != steps):
            p = p - step_size*grad_U(q,sig_inv)
    # make final half step for momentum at the end
    
    p = p - (step_size / 2)*grad_U(q,sig_inv)
    
    # negate the momentum to ensure the Metropolis proposal is symmetric
    
    p = -p
    
    # Evaluate potential and kinetic energies at start and end of 
    # trajectory. 

    #Accept or reject state at end of trajectory, returning either the 
    # postion at the end of the trajectory or the intial postion
    delta_ham = ham(q_current, p_current, sig_inv) - ham(q,p,sig_inv)
    q_new     = accept_reject(delta_ham,q,q_current)
    return q_new
def accept_reject(delta_ham,q, q_current):
    '''Performs the accept reject step - delta_ham is an d x 1 array '''
  
    accept_prob = np.exp(delta_ham)
    u           = np.random.uniform(0,1)
    if (u < accept_prob):
        return q # accept
    else:
        return q_current # reject

def plot_trajectories(q_chain):
    plt.figure(1)
    for coord in q_chain:
        plt.plot(coord[0,0],coord[1,0],'b.')
    sigma   = np.array([[1, 0.8],[0.8,1]])
#    x       = np.linspace(-2,2,1000)
#    y       = np.linspace(-2,2,1000)
#    xv, yv  = np.meshgrid(x,y)
    x,y = np.mgrid[-2:2:0.02, -2:2:0.02]
    pos = np.dstack((x,y))
    rv  = ss.multivariate_normal(mean = np.array([0, 0]), cov = sigma)
    plt.contourf(x,y,rv.pdf(pos))
        
np.random.seed(1)
q_chain     = []
no_samples  = 10000
step_size   = 0.18
leap_step   = 25
q_current   = np.array([[-1.5],[-1.55]])
q_chain.append(q_current)
for i in range(no_samples):
    # runs the leap frog and acc/reject step 
    # and returns the new q_current
    q_new     = hamiltonian_monte_carlo(step_size,leap_step,q_current,i)
    q_current = q_new
    q_chain.append(q_current)
plot_trajectories(q_chain)

print(q_chain)
