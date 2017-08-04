#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:00:58 2017

@author: bradley
"""

import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt 

def gaussian_log_posterior(x, cov, grad = False):
    """Evaluate the unormalized log posterior from a zero-mean
    Gaussian distribution, with the specifed covariance matrix
    
    Parameters
    ----------
    x : torch.autograd.Variable
    Sample ~ target distribution
    covariance : torch.autograg.Variable N x N 


    Returns
    -------
    logp : float, log(exp(-x.T * A * x)) "Log of Normal"
        Unormalized log p(x)
    dlogp_dx : float: Gradient of the log. 
    """
    covariance_inverse = Variable(torch.inverse(cov).float(), requires_grad = False)
    xAx = -0.5*x.mm(covariance_inverse).mm(torch.transpose(x,0,1))
    if grad:
        xAx.backward()
        dlogp_dx = x.grad.data
        return dlogp_dx 
    else:
        return xAx.data



def gaussian_log_posterior_correlated(x, ndim, correlation=0.8, grad = False):
    '''
     Creates a positve definite and symmetric covaraince matrix.
     For any square matrix the following hold: 
     M = M_{s} + M_{a} 
     M_{s} = 0.5*(A  + A.T)
     M_{a} = 0.5*(A - A.T)
     
     Parameters
     ----------
     x           -  torch.autograd.Variable \mathbb{R}^{1,d}
     correlation - Specifies the correlation of the off diagonals
     grad        - bool - specifies if gradient data is required 
    '''
    covariance  = correlation * np.ones((ndim,ndim))
    # the diagonals have standard deviations of sqrt(var)
    var  = 1.0
    np.fill_diagonal(covariance, var)
    covariance = torch.from_numpy(covariance)
    
    return gaussian_log_posterior(x, covariance, grad)

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

def main():
    num_samples = 100

    # Number of dimensions for samples
    ndim  = 10
#    sigma = 0.8
    sample1 = torch.Tensor(num_samples, ndim)
    # Really should put this into a class
#    global cov
#    cov = np.ones((ndim,ndim))
#    np.fill_diagonal(cov, sigma)
#    cov =torch.from_numpy(cov)
#    global cov_pos
#    cov_pos    = 0.5 * (cov + cov.T)
#    cov_pos    = torch.from_numpy(cov_pos)
    print("Drawing from a correlated Gaussian...")
    initial_x = Variable(torch.randn(1, ndim), requires_grad  =True)
    sample1[0] = initial_x.data
    for i in range(num_samples-1):
        sample = hmc(initial_x, 
                log_posterior=gaussian_log_posterior_correlated, 
                step_size = torch.min(torch.Tensor(1).uniform_(0.14,0.18)), 
                num_steps = 29,
                ndim      = ndim)
        sample1[i+1] = sample.data
    
    sampl1np = sample1.numpy()
    print(sampl1np)
    sam1mean   = sampl1np.mean(axis = 0)
    samp1_var  =  np.cov(sampl1np.T)
    print()
    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean 1: ', sam1mean)
    print('empirical_cov 1 :\n', samp1_var)
#    print('****** TRUE MEAN AND COV *****')
#    print('true mean: ', np.zeros((1,ndim)))
#    print('true cov: ' , cov.numpy())
#    plt.figure(1)
#    for i in sampl1np:
#        plt.plot(i[0],i[1],'r.')

main()