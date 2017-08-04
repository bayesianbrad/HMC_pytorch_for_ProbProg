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
    x : tf.Variable
        Sample ~ target distribution
    covariance : tf.Variable N x N 
        Covariance matrix for N-dim Gaussian

        For diagonal - [[sigma_1^2, 0], [0, sigma_2^2]]

    Returns
    -------
    logp : float
        Unormalized log p(x)
    """
    covariance_inverse = Variable(torch.inverse(covariance).float(), requires_grad = False)
    xAx = 0.5*x.mm(covariance_inverse).mm(torch.transpose(x,0,1))
    if grad:
        xAx.backward()
        dlogp_dx = x.grad.data
        return dlogp_dx 
    else:
        return xAx.data


def gaussian_log_posterior_diagonal(x, sigma=1, grad = False):
    covariance =  np.array([[sigma, 0], [0, sigma]])
    covariance = torch.from_numpy(covariance)

    return gaussian_log_posterior(x, covariance, grad)

def gaussian_log_posterior_correlated(x, correlation=0.3, grad = False):
    covariance = np.array([[1.0, correlation],[correlation, 1.0]])
    covariance = torch.from_numpy(covariance)
    return gaussian_log_posterior(x, covariance, grad)


def main():
    num_samples = 10000

    # Number of dimensions for samples
    ndim = 2
    sample1 = torch.Tensor(num_samples, ndim)
    sample2 = torch.Tensor(num_samples, ndim)
    print("Drawing from a correlated Gaussian...")
    initial_x = Variable(torch.randn(1, ndim), requires_grad  =True)
    sample1[0] = initial_x.data
    sample2[0] = initial_x.data
    for i in range(num_samples-1):
        sample = hmc(initial_x, 
                log_posterior=gaussian_log_posterior_correlated, 
                step_size=torch.min(torch.Tensor(1).uniform_(0.01,0.09)), 
                num_steps=5)
        sample1[i+1] = sample.data

    print("Drawing from a diagonal Gaussian...")
    
    for i in range(num_samples-1):
        sample = hmc(initial_x, 
                log_posterior=gaussian_log_posterior_correlated, 
                step_size=0.07, 
                num_steps=20)
        sample2[i+1] = sample.data
#    print('These are sample1 ', sample1)
#    print('These are sample2 ', sample2)
    sampl1np = sample1.numpy()
    sampl2np = sample2.numpy()
    
    sam1mean   = sampl1np.mean(axis = 0)
    samp1_var  =  np.cov(sampl1np.T)
    sam2mean   = sampl2np.mean(axis = 0)
    samp2_var  =  np.cov(sampl2np.T)
    print('****** EMPIRICAL MEAN/COV USING HMC for Diagonal ******')
    print('empirical mean 1: ', sam1mean)
    print('empirical_cov 1 :\n', samp1_var)
    print('****** EMPIRICAL MEAN/COV USING HMC for Corrrelated ******')
    print('empirical mean 2: ', sam2mean)
    print('empirical_cov 2 :\n', samp2_var)
    plt.figure(1)
    for i in sampl1np:
        plt.plot(i[0],i[1],'r.')
    plt.figure(2)
#    for i in sampl2np:
#        plt.plot(i[0],i[1],'r.')
main()