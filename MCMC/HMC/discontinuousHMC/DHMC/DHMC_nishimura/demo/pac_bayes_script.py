import sys
sys.path.insert(0, '..')
from dhmc.dhmc_sampler import DHMCSampler
from other_samplers.mcmc_diagnostic import batch_ess
import numpy as np
import matplotlib.pyplot as plt
import math


y = np.load('../data_and_posterior/secom_outcome.npy')
X = np.load('../data_and_posterior/secom_design_matrix.npy') # With intercept.
n_param = X.shape[1]
n_disc = n_param # No conditional density is smooth.

# Define functions to compute the posterior density

def f(theta, req_grad=True):
    """
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
    """
    
    logp = 0
    grad = np.zeros(len(y))
    
    # Contribution from the prior.
    logp += - np.sum(theta ** 2) / 2
    
    # Contribution from the likelihood.
    y_hat = np.dot(X, theta)
    loglik = np.count_nonzero(y * y_hat > 0)
    logp += loglik
    
    aux = (loglik, y_hat)
    return logp, np.zeros(len(theta)), aux

def f_update(theta, dtheta, j, aux):
    """
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
    """
    
    loglik_prev, y_hat = aux
    y_hat = y_hat + X[:,j] * dtheta
    
    logp_diff = (theta[j] ** 2 - (theta[j] + dtheta) ** 2) / 2
    
    # Contribution from the likelihood.
    loglik = np.count_nonzero(y * y_hat > 0)
    logp_diff += loglik - loglik_prev
    
    aux_new = (loglik, y_hat)
    return logp_diff, aux_new

# Intial state for MCMC
intercept0 = np.log(np.mean(y == 1) / (1 - np.mean(y == 1)))
beta0 = np.zeros(X.shape[1])
beta0[0] = intercept0
theta0 = beta0

# Testing gradients
scale = np.ones(n_param)
dhmc = DHMCSampler(f, f_update, n_disc, n_param, scale)
dhmc.test_cont_grad(theta0, sd=.01, n_test=10);
_, theta, logp_fdiff, logp_diff = \
    dhmc.test_update(theta0, sd=10, n_test=100)

# Run DHMC
seed = 1
n_burnin = 10 ** 2
n_sample = 10 ** 3
dt = .3 * np.array([.7, 1]) 
nstep = [20, 30] 
samples, logp_samples, accept_prob, nfevals_per_itr, time_elapsed = \
    dhmc.run_sampler(theta0, dt, nstep, n_burnin, n_sample, seed=seed)
    
dhmc_samples = samples[n_burnin:, :]

# compute ESS and plot them
ess_dhmc = batch_ess(dhmc_samples, normed=False)
index_sort = np.argsort(ess_dhmc)

plt.figure(figsize=(14, 5))
plt.rcParams['font.size'] = 18

plt.subplot(1, 2, 1)
plt.plot(ess_dhmc[index_sort[:100]])
plt.ylabel('ESS')
plt.xlabel('Param index (sorted)')
plt.title('ESS of 100 worst mixing params')
plt.ylim(0, 1.05 * np.max(ess_dhmc[index_sort[:100]]))

plt.subplot(1, 2, 2)
plt.plot(dhmc_samples[:, index_sort[:3]])
plt.ylabel('Param values')
plt.xlabel('MCMC iteration')
plt.title('Traceplot of 3 worst mixing params')

plt.show()

