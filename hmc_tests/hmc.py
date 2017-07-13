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
import numpy as np
from matplotlib import pyplot as plt

def euler_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
    # Assumes mass is constant for all particles
    p_prev = init_p
    q_prev = init_q
    
    
    for i in range(steps):
        # Plot phase space
        plt.figure(1)
        plt.plot(q_prev,p_prev,'r*')
        # Calculate next p and q values
        p_next = p_prev  - step_size * q_prev
        q_next = q_prev  + (step_size * p_prev)/m
        
        # Update p and q                   
        p_prev = p_next
        q_prev = q_next 
        print(q_prev, p_prev)
        plt.hold(True)

def modified_euler_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
        # Assumes mass is constant for all particles
    p_prev = init_p
    q_prev = init_q
    
    
    for i in range(steps):
        # Plot phase space
        plt.figure(2)
        plt.plot(q_prev,p_prev,'r')
        # Calculate next p and q values
        p_next = p_prev  - step_size * q_prev
        q_next = q_prev  + (step_size * p_next)/m
        
        # Update p and q                   
        p_prev = p_next
        q_prev = q_next 
        print(q_prev, p_prev)
        plt.hold(True)

def leap_frog_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
            # Assumes mass is constant for all particles
    p_prev = init_p
    q_prev = init_q
    
    for i in range(steps):
        # Plot phase space
        plt.figure(3)
        plt.plot(q_prev,p_prev,'r*')
        
        # leapfrog step added
        p_leapfrog = p_prev - (step_size/2)*q_prev
        q_next     = q_prev + (step_size*p_leapfrog)/m
        p_next     = p_leapfrog - (step_size/2)*q_next
        
        # Update p and q                   
        p_prev = p_next
        q_prev = q_next 
        print(q_prev, p_prev)
        plt.hold(True)     

def grad_U(q, inv_cov):
    '''Takes a postion and a defined poptential and returns the derivative
        assumes has dimensions [2 x 1]'''
   return np.dot(np.transpose(q), inv_cov)
    # Returns gradient of potential evaluated at q
    # Things to note:
        # q can be a vector.
    

def Upotential(q, inv_cov):
    # the potential function
    # TO COMPLETE
    # Returns the value of the potential evaluated at q
    # Things to note:
    return np.dot(np.dot(np.transpose(q), inv_cov),q)
def Kkinetic(p):
    return  np.sum(np.dot(p,p))/2 
def grad_K(q):
    ''' The gradient of the kinetic energy. Takes vector input q, composed
        of the componets of the "postions" our varibles of interest  '''\
    return p
def hamiltonian_monte_carlo(step_size, steps, current_q, inv_cov, iternum):
    ''' Performs one interations of the HMC, and perfroms a trajectory with
        L leapfrog steps. The step returns the value of the Hamiltonian and 
        the new proposed q value (may be the same if rejected).  The p's are
        generated only within the function and then discarded.'''
    q = q_current
    # draws from a normal N(0,1)
    if (iternum == 0):
        p = np.array([-1, 1])
    else:
        p = np.random.randn(len(q))  
    
    current_p = p
    
    # make half step for momentum at the beginning
    p = p - (step_size/2)*grad_U(q,inv_cov)
    
    # Alternate fulls teps for position and momentum
    
    for i in range(L):
        # make half step for postion
        q = q + (step_size) * grad_K(p)
        # make full step for postion if i < steps
        
        if (i != L):
            p = p - step_size*grad_U(q)
    # make final half step for momentum at the end
    
    p = p - (step_size / 2)*grad_U(q)
    
    # negate the momentum to ensure the Metropolis proposal is symmetric
    
    p = -p
    
    # Evaluate potential and kinetic energies at start and end of 
    # trajectory. 
    
    current_U  = Upotential(q_current)
    proposed_U = Upotential(q)
    current_K  = Kkinetic(current_p)
    proposed_K = Kkinetic(p)    

    #Accept or reject state at end of trajectory, returning either the 
    # postion at the end of the trajectory or the intial postion
    
    uniform_con     = np.random.uniform(0,1)
    acceptance_prob = np.exp(current_U - proposed_U + current_K - proposed_K) 
    if (uniform_con < acceptance_prob):
        return q # accept
    else:
        return current_q # reject
main():
    cov       = np.array([[1 , 0.95], [0.95,1]])
    q_init    = np.array([-1.5,-1.55])
    L         = 25
    step_size = 0.25
    Ham_vals  = np.zeros(2,1)
    q_values  = q_init
    # Only for plotting
    p_values  = np.array([-1, 1])
    nsamples  = 50
    t         = 0 
    while (t < nsamples):
        t += 1
        