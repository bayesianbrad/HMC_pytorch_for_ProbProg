#!/usr/bin/env python3
"""
Created on Wed Jul 26 14:59:28 2017

# -*- coding: utf-8 -*-
@author: bradley

Notes:
    

"""


import numpy as np
import torch
from torch.autograd import Variable

class SizeError(Exception):
    pass

def findreasonable_epsilon(theta,dim):
    '''A function that implements algorithm 4 from (Hoffman and Gelmans 2014) 
    a heuristic approach for finding the right step size \epsilon. 
    
    theta - a slice of a torch.Variable -  our parameter of interest 
    theta (label x) is constructed as follows:
                [ x11 x12 .... x1D]         [ p11 p12 .... p1D]
                [ x21   ....   x2D]         [ p21   ....   p2D]
   theta =      [ :    ....      :]    p   =[ :   ....      : ] 
                [ :   ....       :]         [ :   ....       :]
                [xN1 ....      xND]         [pN1 ....      pND]
                
    
    this function deals with a slice theta[i][:] = [xi1,...,xiD]
    We have to make n calls of this function for each batch - POTENTIALLY SLOW
    
    dim    - the number of dimensions we wish to sample from
    '''
    
    # for each row vector in theta, representing a particle at a point in 
    # space. We would like to associate a good intial starting value
    stepsize = tensor.ones(theta.size(0))
    
    # we can switch the option volitile - True, rather than require_grad.
    # much fast for inference. See pytorch documentation. 
    
    p_init   = torch.randn(theta.size())
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
        
        theta     -   torch.Variable \mathbb{R}^{1 x D} type tensor float
        p         -   torch.Variable \mathbb{R}^{1 x D} type tensor float        stepsize  -   scalar 
        
    '''
    if isinstance(p, np.ndarray):
        p     = torch.from_numpy(p).float()
        
    # first half step momentum - hopefully can replace with pytorch command.
    p        = p + 0.5*stepsize*potential_fn(theta, gauss = True, grad = True)
    # full step theta
    theta    = theta + stepsize*kinetic_fn(p, gauss = True, grad = True)
    # completing full step of momentum
    p        = p + 0.5*stepsize*potential_fn(theta, gauss = True, grad = True)
    
    return theta, p
        

def metropolis_accept_reject(theta_next,theta_current, p_next, p_current):
    ''' Performs the metropolis accept/ reject step by calculating the
    hamiltonians, exponentiating and checks the required MH condition. 
    If satified the new proposed steps are returned, else the next step
    has the same initial conditions as the previous step.
    
    theta_next        - the proposed position
    theta_current     - the current position
    p_next            - the proposed momentum
    p_current         - the current momentum
    
    returns *to update*
    
    '''
    
    ham_proposed = Hamiltonian(theta_next, p_next)
    ham_current  = Hamiltonian(theta_current, p_current)
    energy_diff  = ham_proposed - ham_current
    accept_prob  = torch.min(torch.ones(1,1),torch.exp(energy_diff))
    # sample from uniform dist
    u            = torch.rand(1,1)
    # will return a  tensor of bool type : 1 if proposal accept, 0 if reject
    accept       = (accept_prob - u >=0)
    
    # the minus is to preserve symmetry
    if (accept[0]):
        return theta_next, -p_next,accept_prob
    else:
        return theta_current, -p_current, accept_prob
     
def chmc_with_dualavg(theta_init, delta, simulationlength, no_samples, no_adapt_iter, batch = 1):
    '''This function implements algorithm 5 of the NUTS sampler 
    Hoffman and Gelman 2014, to create the optimal value for the stepsize
    using stochastic optimization.
    
    thtea_init          - initial theta, \mathbb{R}^{1 \times D}
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
    batch               - Note: All code is currently suited for just one 
                          batch
    returns 
    
    theta_new
    p__new
    theta              - All trajectories calculated for all runs - for 
                         calculating required statistics and plotting (in 3d
                         or less)
    
    '''
    # initial parameters
    
    stepsize            = findreasonable_epsilon(theta_init)
    mu                  = torch.log(10*stepsize_init)
    stepsize_avg        = torch.ones(1,1)
    H_avg               = torch.zeros(1,1)
    gamma               = 0.05 * torch.ones(1,1)
    t_0                 = 10 * torch.ones(1,1)
    kappa               = 0.75 * torch.ones(1,1)
    theta_prev          = theta_init
    # To store all trajectories
    theta               = torch.Tensor(no_samples, theta_init.size(1))
    theta[0][:]         = theta_init
    
    for i in range(1,no_samples):
        p_init        = torch.randn(theta_init.size())
        # Everything starts with intial values and will be modified 
        # accordingly
        theta_current = theta_prev.clone()
        p_tilde       = p_init
        # the max operation only returns a float
        ratio         = torch.round(torch.div(simulationlength,stepsize_current))
        L             = torch.max(torch.ones(1,batch), ratio)
        
        # simulate trajectories.
        for j in range(0,L):
            theta_next, p_next = leapfrog(theta_current, p_tilde, stepsize_current)
        
               
        # perform acceptance step
        theta_next, p_next, accept_prob = metropolis_accept_reject(theta_next, theta_current,\
                                                 p_next, p_tilde)
        # update the trajectory tensor (our drawn samples)
        theta[i][:]   = theta_next
        
        # Adapt parameters 
        
        if i <= no_adapt_iter:
            itensor = i * torch.ones(1,1)
            const   = 1 / (itensor + t_0)
            H_avg   = (1 - const)*H_avg + const* (delta -  accept_prob)
            
            #log_newstepsize
            stepsize      = torch.exp(mu - torch.div(torch.sqrt(itensor), gamma)*H_avg)
            temp          = torch.pow(itensor, -kappa)
            stepsize_avg  = torch.exp(temp*torch.log(stepsize) + (1- temp)*torch.log(stepsize_avg))
        else:
            stepsize = stepsize_avg
    return theta

def potential_fn(theta,dim, grad = False):
    '''Calulates the potential energy. Uses the pytorch back end to 
     automatically calcualte the gradients. 
     All variables that go into the potential_fn will have to be declared
     as torch.autograd.Variable(<variable>, requires_grad = <bool>)
     Where only if the gradient of the variable is required do you
     set the bool to True, else set to False. 
     
     theta     - Variable object in pytorch.\mathbb{R}^{1 \times D}
     grad      - bool - If true returns gradient w.r.t theta.
     dim       - dimension of system 
     
     returns  U(\theta) if grad = False
          or  dU_dtheta if grad = True
     
     ************* N-dim Gaussian implemented here for testing *************
     '''
     # convert to torch float, if np array .
    if isinstance(theta, np.ndarray):
       theta = torch.from_numpy(theta).float()
    
    # convert theta to a variable.
    theta    = Variable(theta, requires_grad = True)
#==============================================================================
#     May want to move the construction of the mean and cov outside of
#     the function. It will be much cleaner and much more useful fo0r 
#     testing. 
#==============================================================================
    # randn draws from N(0,1) use .normal_(mean = <mean>, std = <std> )
    mu   = Variable(torch.randn(dim), requires_grad = False) 
    cov  = torch.randn(dim,dim)
    # ensures postive definite and symmetric
    cov  = (cov + torch.transpose(cov, dim0 = 0, dim1 = 1)) / 2
    for i in range(dim):
        cov[i][i]  = 1
    cov_inv = Variable(torch.inverse(cov), requites_grad = False)
    
    # calcualte the potential 
    potential = 0.5 * (theta - mu).mm(cov_inv).mm(torch.transpose(theta - mu),0,1))
    if grad:
        potential.backward()
        dU_dtheta = theta.grad.data
        # zero gradients
        theta.grad.data.zero()
        ##*#****$**$*$*$*$$**$
        # try:except, only for testing
        ##*#****$**$*$*$*$$**$
        # ensure size of gradient \equiv to size of theta
        try:
            boolean = theta.size() == dU_dtheta.size())
            return dU_dtheta
        except SizeError as e:
            print("The size of the gradient and theta do not match")
    else:
        return potential


def kinetic_fn(p, M_inv, gauss  = True, laplace = False,  grad = False):
    '''Implements the given kinetic energy of the system. Automatically 
    calulates gradients if required. 
    
    p       -  \mathbb{R}^{1 \times D} dtype = float32
    M_inv   - Is a positive symmetric, diagonal matrix. Each diag represents
              M_inv_{ii} represents the 'mass' component p_{i} dype = float32
    gauss   - bool - if True calculates the stadard gaussian K.E.
                     K(p) = 0.5*M^{-1}* p*p  Does elementwise multiplication
    laplace - bool - if True calculates the Laplace momentum instead
                     K(p) = m^{-1} | p | 
    grad    - bool - if True calcules gradient dk/dp
    
    returns
    
    K(p) or dk_dp
    '''
    # checks whether object is numpy array and converts to torch Float
    # tensor
    if isinstance(M_inv, np.ndarray):
        M_inv = torch.from_numpy(M_inv).float()
    if isinstance(p, np.ndarray):
        p     = torch.from_numpy(p).float()
    # create torch.autograd.Variable objects that can be differentiated.  
    
   
    p     = Variable(p, requires_grad = True)
    M_inv = Variable(M_inv, requires_grad = False)
    if gauss:
        K =  p.mm(M_inv).mm(torch.transpose(p,0,1))
        if grad:
            K.backward()
            dk_dp = p.grad.data
            # zero gradients 
            p.grad.data.zero_()
            return dk_dp
        else:
            return K
    else:
         K = torch.mm(torch.abs(p),M_inv) 
         if grad:
             return 0#only makes sense in the discrete case as derivative of |x| is undefined
         else:
             return K


#    Needed for when running the sampler
#  
#    print('****** TARGET VALUES ******')
#    print('target mean:', mu)
#    print('target cov:\n', cov)
#
#    print('****** EMPIRICAL MEAN/COV USING HMC ******')
#    print('empirical mean: ', samples.mean(axis=0))
#    print('empirical_cov:\n', numpy.cov(samples.T))
#
#    print('****** HMC INTERNALS ******')
#    print('final stepsize', sampler.stepsize.get_value())
#    print('final acceptance_rate', sampler.avg_acceptance_rate.get_value())
