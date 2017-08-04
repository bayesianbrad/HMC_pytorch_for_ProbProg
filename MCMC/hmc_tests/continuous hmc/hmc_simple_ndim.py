#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:44:43 2017

@author: bradley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:01:50 2017

@author: bradley
"""

import torch
from torch.autograd import Variable
import numpy as np

def kinetic_energy(velocity):
    """Kinetic energy of the current velocity (assuming a standard Gaussian)
        (x dot x) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}
    Parameters
    ----------
    velocity : torch.autogram.Variable
        Vector of current velocity
    Returns
    -------
    kinetic_energy : float
    """
    return 0.5 * torch.dot(velocity, velocity)

def hamiltonian(position, velocity, energy_function,ndim):
    """Computes the Hamiltonian  given the current postion and velocity
    H = U(x) + K(v)
    U is the potential energy and is = -log_posterior(x)
    Parameters
    ----------
    position : tf.Variable
        Position or state vector x (sample from the target distribution)
    velocity : tf.Variable
        Auxiliary velocity variable
    energy_function
        Function from state to position to 'energy'
         = -log_posterior
    Returns
    -------
    hamitonian : float
    """
    U = energy_function(position,ndim) 
    T = kinetic_energy(velocity)
    return U + T

def leapfrog_step(x0, 
                  v0, 
                  log_posterior,
                  step_size,
                  num_steps,
                  ndim):
    '''Performs the leapfrog steps of the HMC for the specified trajectory
    length, given by num_steps'''

    # Start by updating the velocity a half-step
    v = v0.data + 0.5 * step_size * log_posterior(x0, ndim, grad = True)
    # Initalize x to be the first step
    x0.data = x0.data + step_size * v
    
    x0.grad.data.zero_()
    for i in range(num_steps):
        # Compute gradient of the log-posterior with respect to x
        # Update velocity
        v = v + step_size * log_posterior(x0, ndim, grad = True)

        # Update x
        x0.data = x0.data + step_size * v
        x0.grad.data.zero_()

    # Do a final update of the velocity for a half step
    
    v = v + 0.5 * step_size * log_posterior(x0,ndim,grad = True)
   
    # return new proposal state
    return x0, v

def hmc(initial_x,
        step_size, 
        num_steps, 
        log_posterior,
        ndim):
    """Summary
    Parameters
    ----------
    initial_x : torch.autograd.Variable
        Initial sample x ~ p
    step_size : float Step-size in Hamiltonian simulation 
    num_steps : int  Number of steps to take in leap frog
    log_posterior : str
        Log posterior (unnormalized) for the target distribution
        takes ndim, grad(bool)
    Returns
    -------
    sample : 
        Sample ~ target distribution
    """
    v0 = Variable(torch.randn(initial_x.size()),requires_grad = False)
    x, v = leapfrog_step(initial_x,
                      v0, 
                      step_size=step_size, 
                      num_steps=num_steps, 
                      log_posterior=log_posterior,
                      ndim = ndim)

    orig = hamiltonian(initial_x, v0.data, log_posterior,ndim)
    current = hamiltonian(x, v, log_posterior,ndim)
    alpha = torch.min(torch.exp(orig - current))
    p_accept = min(1,alpha)
    if p_accept > np.random.uniform():
        return x
    else:
        return initial_x
