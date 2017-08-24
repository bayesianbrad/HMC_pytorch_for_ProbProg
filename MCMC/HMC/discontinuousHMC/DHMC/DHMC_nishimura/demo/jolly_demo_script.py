#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:44:10 2017

@author: bradley
g"""

import sys
sys.path.insert(0, '..')
from dhmc.dhmc_sampler import DHMCSampler
import numpy as np
import math
import matplotlib.pyplot as plt
from data_and_posterior.jolly_seber_model import f, f_update
'''
def f(theta, req_grad=True):  
    
    Computes the log posterior density and its gradient. 
    
    Params:
    ------
    theta : ndarray
    req_grad : bool
        If True, returns the gradient along with the log density.
    
    Returns:
    -------
    logp : float
    grad : ndarray
    aux : Any
        Any computed quantities that can be re-used by the 
        subsequent calls to the function 'f_updated' to save on
        computation.

    
def f_update(theta, dtheta, j, aux):  
    Computes the difference in the log conditional density 
    along a given parameter index 'j'.
    
    Params:
    ------
    theta : ndarray
    dtheta : float
        Amount by which the j-th parameter is updated.
    j : int
        Index of the parameter to update.
    aux : Any
        Computed quantities from the most recent call to functions
        'f' or 'f_update' that can be re-used to save on computation.
    
    Returns:
    -------
    logp_diff : float
    aux_new : Any

The parameters "p", "phi", "U" are concatenated into a 1-d array for
running DHMC. The dictionary "index" stores the linear indices used 
internally by 'f' and 'f_update'.
 '''
from data_and_posterior.jolly_seber_model import pack_param, unpack_param, index
 # Number of continuous and discrete parameters.
from data_and_posterior.jolly_seber_model \
    import n_param, n_disc, n_cont

# pick intial state for MCMC
phi0 = .8 * np.ones(len(index["phi"]))
p0 = .15 * np.ones(len(index["p"]))
U0 = 500 * np.ones(len(index["U"]))
theta0 = pack_param(p0, phi0, U0)

# intialise the DHMC sampler and the test outputs of f and f_update
scale = np.ones(n_param)
dhmc = DHMCSampler(f, f_update, n_disc, n_param, scale)

dhmc.test_cont_grad(theta0, sd=1, n_test=10);
_, theta, logp_fdiff, logp_diff = \
    dhmc.test_update(theta0, sd=10, n_test=10)

# Run DHMC
seed = 1
n_burnin = 10 ** 2
n_sample = 1 * 20 ** 3
n_update = 10
dt = .025 * np.array([.8, 1])
nstep = [70, 85]

samples, logp_samples, accept_prob, pathlen_ave, time_elapsed = \
    dhmc.run_sampler(theta0, dt, nstep, n_burnin, n_sample, 
                     seed=seed, n_update=n_update)
dhmc_samples = samples[n_burnin:, :]

#Check mixing with traceplots.
p_samples, phi_samples, U_samples, N_samples = \
    unpack_param(dhmc_samples)
plt.figure(figsize=(14, 5))
plt.rcParams['font.size'] = 16

plt.subplot(1, 2, 1)
plt.plot(p_samples)
plt.title(r"Traceplot of capture probabilities $p_i$'s")
plt.ylabel(r"$p_i$")
plt.xlabel('MCMC iteration')

plt.subplot(1, 2, 2)
plt.plot(N_samples)
plt.title(r"Traceplot of population sizes $N_i$'s")
plt.ylabel(r"$N_i$")
plt.xlabel('MCMC iteration')

plt.show()