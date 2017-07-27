#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:59:28 2017

@author: bradley

Notes:
    
    Ensure in the potential and energy functions that we are doing element
    wise multiplication and returning R^{1 \times D}
    
    Do we really want that?????
    U(theta) should return a scalar given some vector R^{1 \times D}
    
"""
# alogirthm 4
import numpy as np
import torch
from torch.autograd import Variable


def findreasonable_epsilon(theta):
    '''A function that uses (Hoffman and Gelmans 2014) approach to finding
    the right step size \epsilon. 
    
    theta - a slice of a torch.Variable -  our parameter of interest 
    theta (label x) is constructed as follows:
                [ x11 x12 .... x1D]         [ p11 p12 .... p1D]
                [ x21   ....   x2D]         [ p21   ....   p2D]
   theta =      [ :    ....      :]    p   =[ :   ....      : ] 
                [ :   ....       :]         [ :   ....       :]
                [xN1 ....      xND]         [pN1 ....      pND]
                
    
    this function deals with a slice theta[i][:] = [xi1,...,xiD]
    We have to make n calls of this function for each batch - POTENTIALLY SLOW
    '''
    
    # for each row vector in theta, representing a particle at a point in 
    # space. We would like to associate a good intial starting value
    stepsize = tensor.ones(theta.size(0))
    p_init   = Variable(torch.randn(theta.size()),requires_grad = True)
    # may have to intialise theta_new and p_new as variables with 
    # reqires_grad = True. If properties are not carried forward. 
    
    theta_next, p_next = leapfrog(theta, p_init, stepsize)
    
    # This will return a scalar for each batch. 
    p_joint_next     = cal_joint(theta_next,p_next)
    p_joint_prev     = cal_joint(theta , p_init)
    
    alpha            = p_joint_new / p_joint_prev
   #implement if statement in tensors
   # if we decide to processes all batches at once we can use the following 
   # line:    a_values         = 2*(alpha>0.5).float() - 1 
    a_value           = 2*(alpha>0.5).float() - 1
    truth             = torch.gt(torch.pow(alpha,a),torch.pow(2,-a_values))
    
    # The following while loop, should return a stepsize, as a tensor, with 
    # the optimal starting value. 
    while(truth[0][0]):
        stepsize           = torch.pow(2,a_values) * stepsize
        theta_next, p_next = leapfrog(theta, p_init, stepsize)
        
        p_joint_next     = cal_joint(theta_next,p_next)
        p_joint_prev     = cal_joint(theta , p_init)
        alpha            = p_joint_new / p_joint_prev
        truth            = torch.gt(torch.pow(alpha,a),torch.pow(2,-a_values))

    return stepsize 

def cal_joint(theta_row, p_row):
    '''Takes the one of column of theta and the corresponding column 
    of p, and returns a scalar value of the joint P(\theta, p) at
    the corresponding values. The potential_fn and kinetic_fn both return
    scalars, thus the output is a scalar. But it is a torch.Tensor scalar
    of size (1,1)'''
    return torch.exp(-Hamiltonian(theta_row,p_row))

def Hamiltonian(theta, p):
    ''' Returns the Hamiltonian K(P;theta) + U(theta). The notation ; 
    corresponds to: this function may be dependent on this variable. '''
    return potential_fn(theta) + kinetic_fn(p)

def leapfrog(theta, p, stepsize):
    '''Performs the integrator step, via the leapfrog method, as described in 
        Dune 1987. 
        
        theta     -   torch.Variable \mathbb{R}^{1 x D} type tensor float64
        p         -   torch.Variable \mathbb{R}^{1 x D} type tensor float64
        stepsize  -   scalar 
        
    '''
    # first half step momentum - hopefully can replace with pytorch command.
    p        = p + 0.5*stepsize*grad_potential_fn(theta)
    # full step theta
    theta    = theta + stepsize*grad_kinetic_fn(p)
    # completing full step of momentum
    p        = p + 0.5*stepsize*grad_potential_fn(theta)
    
    return theta, p
        
def chmc_with_dualavg(theta_init, delta, simulationlength, no_samples, no_adapt_iter):
    '''This function implements algorithm 5 of the NUTS sampler 
    Hoffman and Gelman 2014, to create the optimal value for the stepsize
    using stochastic optimization.
    
    thtea_init          - 
    delta               - desired average acceptance probability
    simulation length   - stepsize * L - L is the number of trajectories run
    no_samples          - How many samples that we want to collect
    no_adapt_iter       - Number of iterations after which to stop the adaptation
    H                   - Statistic that describes some aspect of the behivour
                          of an MCMC algo
    h(stepsize)         - Expectation of H, h(stepsize) = \mathbb{E}_{t}[H_{t} | stepsize]
    t_{0}               - A free parameter taht stabilzes the intial 
                          iterations of the algorithm
    gamma               - is free parameter that controls the amount of
                          shrinkage towards mu
    mu                  - freely choosen pt that the iterates stepsize_{t} 
                          are shrunk towards
    kappa               - free parameter between (0.5,1] - mainly to reduce 
                         the computation time to find optimal stepsize
    '''
    # initial parameters
    
    stepsize_init       = findreasonable_epsilon(theta_init)
    mu                  = torch.log(10*stepsize_init)
    stepsize_avg        = torch.ones(1,1)
    H_avg_init          = torch.zeros(1,1)
    gamma               = 0.05
    t_0                 = 10
    kappa               = 0.75
    
    for i in range(1,no_samples):
        p_init     = torch.randn(theta_init.size())
        theta_next = theta_prev
        
    
    
    
#def HMC(theat_init, stepsize, steplength, potential_ft, kinetic_ft, no_samples):
#    '''Performs the integrator step, via the leapfrog method, as described in 
#    Dune 1987. 
#    theta_init    - torch.Variable \mathbb{R}^{N x D}
#    stepsize      - scalar
#    steplength    - scalar
#    potential_ft  - USER_DEFINED
#    kiniteic_ft   - USER_DEFINED
#    no_samples    - scalar
#    for n = 1 to no_samples:
#        