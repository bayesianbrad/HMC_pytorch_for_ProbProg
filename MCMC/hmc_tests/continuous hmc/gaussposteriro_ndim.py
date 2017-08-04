#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:44:06 2017

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
import time
from hamilton_monte_carlo_simple import hmc
from matplotlib import pyplot as plt 

def gaussian_log_posterior(x, covariance, grad = False):
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
    covariance_inverse = Variable(torch.inverse(covariance).float(), requires_grad = False)
    xAx = 0.5*x.mm(covariance_inverse).mm(torch.transpose(x,0,1))
    if grad:
        xAx.backward()
        dlogp_dx = x.grad.data
        return dlogp_dx 
    else:
        return xAx.data


def gaussian_log_posterior_correlated(x, grad = False):
    ''' constructs covariance matrix'''
    # create positve definite and symmetric covaraince matrix.
    # For any square matrix the following hold: 
    # M = M_{s} + M_{a} 
    # M_{s} = 0.5*(A  + A.T)
    # M_{a} = 0.5*(A - A.T)
    return gaussian_log_posterior(x, cov_pos, grad)

def main():
    num_samples = 10

    # Number of dimensions for samples
    ndim  = 2
    sigma = 0.8
    sample1 = torch.Tensor(num_samples, ndim)
    # Really should put this into a class
    global cov
    cov = np.ones((ndim,ndim))
    np.fill_diagonal(cov, sigma)
    global cov_pos
    cov_pos    = 0.5 * (cov + cov.T)
    cov_pos    = torch.from_numpy(cov_pos)
    print("Drawing from a correlated Gaussian...")
    initial_x = Variable(torch.randn(1, ndim), requires_grad  =True)
    sample1[0] = initial_x.data
    for i in range(num_samples-1):
        sample = hmc(initial_x, 
                log_posterior=gaussian_log_posterior_correlated, 
                step_size = 0.18, 
                num_steps = 5)
        sample1[i+1] = sample.data
    
    sampl1np = sample1.numpy()
    print(sampl1np)
    sam1mean   = sampl1np.mean(axis = 0)
    samp1_var  =  np.cov(sampl1np.T)

    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean 1: ', sam1mean)
    print('empirical_cov 1 :\n', samp1_var)
    print('****** TRUE MEAN AND COV *****')
    print('true mean: ', np.zeros((1,ndim)))
    print('true cov: ' , cov_pos)
#    plt.figure(1)
#    for i in sampl1np:
#        plt.plot(i[0],i[1],'r.')

main()