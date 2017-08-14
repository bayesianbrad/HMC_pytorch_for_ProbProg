#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:31:26 2017

@author: bradley
"""

import torch
import numpy as np
from torch.autograd import Variable

def kinetic_fn(p, mom  ='Gauss', grad = False):
    """Kinetic energy of the current momentum (assuming a standard Gaussian)
        (x dot x) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}
    Parameters
    ----------
    p    : torch.autogram.Variable
          Vector of current momentum
    mom  : String representing type of momentum to use
    grad : bool
    Returns
    -------
    kinetic_energy : float
    """
    # mass matrix
    P = Variable(p, requires_grad = True)
    M = Variable(torch.eye(p.size()[1]), requires_grad = False)
    K = 0.5 * P.mm(M).mm(torch.transpose(P,0,1))
    if grad:
        K.backward()
        return P.grad.data
    else:
        return K.data
    
def log_potential_fn(x, cov_inverse, grad = False):
    """Evaluate the unormalized log posterior from a zero-mean
    Gaussian distribution, with the specifed covariance matrix
    
    Parameters
    ----------
    x : torch.autograd.Variable
    Sample ~ target distribution
    cov_inverse : torch.autograg.Variable N x N 


    Returns
    -------
    logp : float, log(exp(-x.T * A * x)) "Log of Normal"
        Unormalized log p(x)
    dlogp_dx : float: Gradient of the log. 
    """
    if not isinstance(x, Variable):
        X = Variable(x, requires_grad = True)
    else:
        X = x
    xAx = -0.5*X.mm(cov_inverse).mm(torch.transpose(X,0,1))
    if grad:
        xAx.backward()
        dlogp_dx = X.grad.data
        return dlogp_dx 
    else:
        return xAx.data
    
class Borg:
    ''' Borg class making class attributes global'''
    _shared_state = {} # Attribute dictionary
    
    def __init__(self):
        # Make it an attribute dictionary
        self.__dict__  = self._shared_state

class HMCsampler(Borg):
    # HMC sampler class
    def __init__(self, **kwargs):
        '''In general the following will be passed to this class:
            x, p0,  n_dim, n_samples, n_vars, count, potential, kinetic, n_vars '''
        Borg.__init__(self)
        # update the attribute dictionaty by inserting a new key-value pair
        self._shared_state.update(kwargs)
    def __str__(self):
        #Returns the attribute disctionary for printing
        return str(self._shared_state)

    def leapfrog_steps(self):
        '''Performs the leapfrog steps of the HMC for the specified trajectory
        length, given by num_steps
        Parameters
        ----------
            x0
            p0
            log_potential
            step_size
            n_steps
            
        Outputs
        -------
            xproposed
            pproposed
        '''
        x0 = self.x0
        p0 = self.p0
        log_potential = self.log_potential
        step_size     = self.stepsize
        kinetic       = self.kinetic
        n_steps       = self.nsteps
        cov_inv       = self.cov_inv

        # Start by updating the momentum a half-step
        p = p0 + 0.5 * step_size * log_potential(x0, cov_inv, grad = True)
        # Initalize x to be the first step
        x0.data = x0.data + step_size * kinetic(p, grad = True)
        # If the gradients are not zeroed then they will blow up. This leads
        # to an exponentiatin kinetic and potential energy. As the positon
        # and momentum increase. 
        x0.grad.data.zero_()
        for i in range(n_steps-1):
            # Compute gradient of the log-posterior with respect to x
            # Update momentum
            p = p + step_size * log_potential(x0, cov_inv, grad = True)
    
            # Update x
            x0.data = x0.data + step_size *  kinetic(p, grad = True)
            x0.grad.data.zero_()
    
        # Do a final update of the momentum for a half step
        
        p = p + 0.5 * step_size * log_potential(x0, cov_inv , grad = True)
        xproposed = x0
        pproposed = p
        # return new proposal state
        return xproposed, pproposed

    def hamiltonian(self,x,p):
        """Computes the Hamiltonian  given the current postion and momentum
        H = U(x) + K(p)
        U is the potential energy and is = -log_posterior(x)
        Parameters
        ----------
        x             :torch.autograd.Variable, we requires its gradient. 
                         Position or state vector x (sample from the target 
                         distribution)
        p             :torch.Tensor \mathbb{R}^{1 x D}. Auxiliary momentum 
                         variable
        log_potential :Function from state to position to 'energy'= -log_posterior
        
        Returns
        -------
        hamitonian : float
        """
        U = self.log_potential(x,self.cov_inv) 
        T = self.kinetic(p)
        return U + T
    
    def acceptance(self):
        '''Returns the new accepted state
        
        Parameters
        ----------
        x = xproposed
        x0  
        p = pproposed
        p0
        
        Output
        ------
        returns sample
        '''
        # get proposed x and p
        x,p     = self.leapfrog_steps()
        orig    = self.hamiltonian(self.x0, self.p0)
        current = self.hamiltonian(x, p)
        alpha   = torch.min(torch.exp(orig - current))
        # calculate acceptance probability
        p_accept = min(1,alpha)
        if p_accept > np.random.uniform():
            # Updates count globally for target acceptance rate
            self.count = self.count + 1
            return x
        else:
            return self.x0
            
    def run_sampler(self):
        ''' Runs the hmc internally for a number of samples and updates
        our parameters of interest internally
        Parameters
        ----------
        n_samples
        burn_in
        
        Output
        ----------
        A tensor of the number of required samples
        Acceptance rate
        
        
        '''
        print("Drawing from a correlated Gaussian...")
        n_samples  = self.nsamples
        n_dim      = self.ndim
        n_vars     = self.nvars
        min_step   = self.min_step
        max_step   = self.max_step
        min_traj   = self.min_traj
        max_traj   = self.max_traj
        burn_in    = self.burn_in
        samples    = torch.Tensor(n_samples,n_dim)
        samples[0] = self.x0.data
        for i in range(n_samples-1):
            temp       = self.acceptance()
            # update the intial value of self.x0 globally
            self.x0    = temp
            samples[i] = temp.data
            # update parameters and draw new momentum
            self.step_size = np.random.uniform(min_step,max_step)
            self.n_steps   = int(np.random.uniform(min_traj,max_traj))
            self.p0        = torch.randn(n_vars, n_dim)
        
        target_acceptance = self.count / (n_samples - 1)
        sampl1np = samples[burn_in:,:].numpy()
        #    print(sampl1np)
        sam1mean   = sampl1np.mean(axis = 0)
        samp1_var  =  np.cov(sampl1np.T)
        print('****** TRUE MEAN/ COV ******')
        print('True mean: ', np.zeros((1, n_dim)))
        print('True cov: ', self.cov)
        print()
        print('****** EMPIRICAL MEAN/COV USING HMC ******')
        print('empirical mean : ', sam1mean)
        print('empirical_cov  :\n', samp1_var)
        print('Average acceptance rate is: ', target_acceptance)
#        return samples[burn_in:, :], target_acceptance
            
def main():
    n_dim       = 100
    n_samples   = 10000
    burnin      = 0
    n_vars      = 1
    minstep     = 0.03
    maxstep     = 0.18
    mintraj     = 5
    maxtraj     = 15
    # Intialise both trajectory length and step size
    step_size   = np.random.uniform(minstep,maxstep)
    n_steps     = int(np.random.uniform(mintraj,maxtraj))
    sigma       = 0.34782378
    correlation = 0.2998388373
    cov         = correlation* np.ones((n_dim,n_dim))
    np.fill_diagonal(cov, sigma)
    cov         = torch.from_numpy(cov)
    cov_inv     = Variable(torch.inverse(cov).float(), requires_grad = False)
    xinit       = Variable(torch.randn(n_vars, n_dim), requires_grad = True)
    pinit       = torch.randn(n_vars, n_dim)
    hmc_sampler = HMCsampler(x0 = xinit,
                     p0   = pinit,
                     ndim = n_dim,
                     nsamples = n_samples,
                     burn_in  = burnin,
                     nvars    = n_vars,
                     nsteps   = n_steps,
                     stepsize = step_size,
                     min_step = minstep,
                     max_step = maxstep,
                     min_traj = mintraj,
                     max_traj = maxtraj,
                     cov      = cov,
                     cov_inv  = cov_inv,
                     count    = 0,
                     log_potential = log_potential_fn,
                     kinetic       = kinetic_fn)
    hmc_sampler.run_sampler()
main()