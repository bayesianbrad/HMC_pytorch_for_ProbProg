#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:11:33 2017

@author: bradley
"""

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

def main():
    global count
    count       = 0
    num_samples = 10000'
    
    

    # Number of dimensions for samples
    ndim       = 100
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
#    intial_x   = torch.randn(1,ndim)
    sample1[0] = initial_x.data
    for i in range(num_samples-1):
        sample,count = hmc(initial_x, 
                log_posterior=gaussian_log_posterior_correlated, 
                step_size = torch.min(torch.Tensor(1).uniform_(0.14,0.18)), 
                num_steps = torch.Tensor(1).uniform_(19,30).int()[0],
                ndim      = ndim,
                count     = count)
        sample1[i+1] = sample.data
    
    sampl1np = sample1.numpy()
#    print(sampl1np)
    sam1mean   = sampl1np.mean(axis = 0)
    samp1_var  =  np.cov(sampl1np.T)
    print()
    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean : ', sam1mean)
    print('empirical_cov  :\n', samp1_var)
    print('Average acceptance rate is: ', count / num_samples)
#    print('****** TRUE MEAN AND COV *****')
#    print('true mean: ', np.zeros((1,ndim)))
#    print('true cov: ' , cov.numpy())
#    plt.figure(1)
#    for i in sampl1np:
#        plt.plot(i[0],i[1],'r.')
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
    if isinstance(x, Variable) == False:
        X = Variable(x, requires_grad = True)
    else:
        X = x
    covariance_inverse = Variable(torch.inverse(cov).float(), requires_grad = False)
    xAx = -0.5*X.mm(covariance_inverse).mm(torch.transpose(X,0,1))
    if grad:
        xAx.backward()
        dlogp_dx = X.grad.data
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
    var  = 1
    np.fill_diagonal(covariance, var)
    covariance = torch.from_numpy(covariance)
    
    return gaussian_log_posterior(x, covariance, grad)

def kinetic_energy(momentum, grad = False):
    """Kinetic energy of the current momentum (assuming a standard Gaussian)
        (x dot x) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}
    Parameters
    ----------
    momentum : torch.autogram.Variable
        Vector of current momentum
    Returns
    -------
    kinetic_energy : float
    """
    # mass matrix
    P = Variable(momentum, requires_grad = True)
    M = Variable(torch.eye(momentum.size()[1]), requires_grad = False)
    K = 0.5 * P.mm(M).mm(torch.transpose(P,0,1))
    if grad:
        K.backward()
        return P.grad.data
    else:
        return K.data
#    return 0.5 * torch.dot(momentum, momentum)

def hamiltonian(position, momentum, energy_function,ndim):
    """Computes the Hamiltonian  given the current postion and momentum
    H = U(x) + K(p)
    U is the potential energy and is = -log_posterior(x)
    Parameters
    ----------
    position        :torch.autograd.Variable, we requires its gradient. 
                     Position or state vector x (sample from the target 
                     distribution)
    momentum        :torch.Tensor \mathbb{R}^{1 x D}. Auxiliary momentum 
                     variable
    energy_function :Function from state to position to 'energy'= -log_posterior
    
    Returns
    -------
    hamitonian : float
    """
    U = energy_function(position,ndim) 
    T = kinetic_energy(momentum)
    return U + T

def leapfrog_step(x0, p0,log_posterior, step_size, num_steps,  ndim):
    '''Performs the leapfrog steps of the HMC for the specified trajectory
    length, given by num_steps'''

    # Start by updating the momentum a half-step
    p = p0 + 0.5 * step_size * log_posterior(x0, ndim, grad = True)
    # Initalize x to be the first step
    x0.data = x0.data + step_size * kinetic_energy(p, grad = True)
    # If the gradients are not zeroed then they will blow up. This leads
    # to an exponentiatin kinetic and potential energy. As the positon
    # and momentum increase. 
    x0.grad.data.zero_()
    for i in range(num_steps-1):
        # Compute gradient of the log-posterior with respect to x
        # Update momentum
        p = p + step_size * log_posterior(x0, ndim, grad = True)

        # Update x
        x0.data = x0.data + step_size *  kinetic_energy(p, grad = True)
        x0.grad.data.zero_()

    # Do a final update of the momentum for a half step
    
    p = p + 0.5 * step_size * log_posterior(x0,ndim,grad = True)
   
    # return new proposal state
    return x0, p

def hmc(initial_x, step_size, num_steps,log_posterior, ndim, count):
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
    p0 = torch.randn(initial_x.size())
    x, p = leapfrog_step(initial_x,
                      p0, 
                      step_size=step_size, 
                      num_steps=num_steps, 
                      log_posterior=log_posterior,
                      ndim = ndim)

    orig = hamiltonian(initial_x, p0, log_posterior,ndim)
    current = hamiltonian(x, p, log_posterior,ndim)
    alpha = torch.min(torch.exp(orig - current))
    p_accept = min(1,alpha)
    if p_accept > np.random.uniform():
        count = count + 1
        return x, count
    else:
        return initial_x,count


main()