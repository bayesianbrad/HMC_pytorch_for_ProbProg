#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:31:26 2017

@author: bradley
"""

import torch
import numpy as np
from torch.autograd import Variable
import scipy.stats as ss
from Utils.core import VariableCast
import math

np.random.seed(1234)
torch.manual_seed(1234)


class KEnergy():
    ''' A basic class that implements kinetic energies and computes gradients
    Methods
    -------
    gauss_ke          : Returns KE gauss
    laplace_ke        : Returns KE laplace

    Attributes
    ----------
    p    - Type       : torch.Tensor, torch.autograd.Variable,nparray
           Size       : [1, ... , N]
           Description: Vector of current momentum

    M    - Type       : torch.Tensor, torch.autograd.Variable, nparray
           Size       : \mathbb{R}^{N \times N}
           Description: The mass matrix, defaults to identity.

    '''
    def __init__(self, M = None):

        if M is not None:
            if isinstance(M, Variable):
                self.M  = VariableCast(torch.inverse(M.data))
            else:
                self.M  = VariableCast(torch.inverse(M))
        else:
            self.M  = VariableCast(torch.eye(p.size()[1])) # inverse of identity is identity


    def gauss_ke(self,p, grad = False):
        '''' (p dot p) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}'''
        self.p = VariableCast(p)
        P = Variable(self.p.data, requires_grad=True)
        K = 0.5 * P.mm(self.M).mm(torch.transpose(P, 0, 1))
        if grad:
            return self.ke_gradients(P, K)
        else:
            return K
    def laplace_ke(self, p, grad = False):
        self.p = VariableCast(p)
        P = Variable(self.p.data, requires_grad=True)
        K = torch.sign(P).mm(self.M)
        if grad:
            return self.ke_gradients(P, K)
        else:
            return K
    def ke_gradients(self, P, K):
        return torch.autograd.grad([K], [P], grad_outputs=torch.ones(P.size()))[0]

class Foppl_reformat(object):
    '''
    Takes an object that is the output of FOPPL and creates an object of the given joint.

    Methods
    ----------
    Need sometihng to extract the keys and values
    turn parameters of interest into Variables with requires grad
    calc the whole logjoint
    calc the derivative, w.r.t to the inputs and outputs




    Attributes
    -------
    FOPPL_out   - Type        : python dictionary keys  : type str < Name of distribution >,
                                                 values : type torch.Tensor, Variable, ndarray <parameter of interest>
                  Size        :
                  Description : Contains the FOPPL output and provides all the information to construct the log_joint.
                                Such as: the probability distribution, a str, and the required parameters of interest
                                for each distribution
    '''

    def __init__(self, FOPPL_out):
        self.Fout = FOPPL_out

    def Variable_updater(self):
        for k, v in self.Fout:
            v = VariableCast(v, requires_grad=True)


# class LogPotentialCts():
#     """ Takes a joint density and creates an object that
#
#
#     Methods
#     -------
#
#     calc_grad  - calculates the gradients of the joint
#     calc_pot   - calculates the potential evaluated at the proposed parameter of interested points, generated
#                           via the leapfrog integrator
#     Attributes
#     ----------
#     joint P(x,y) - Type       : torch.Variable
#                    Size       :
#                    Description: Is the joint distribution, where the joint has been constructed from the FOPPL
#                                 output, via the distribution class module
#
#     params       - Type       : python list of Variables. The data attributes must all be the same size.
#                    Size       :
#                    Description: A python list of the variables of interest. The latent parameters etc. The object, will
#                                 have its gradients evaluated with respect to that.
#
#
#     """
#     def __init__(self, joint, params):
#         self.joint  = torch.log(joint)
#         self.params = params
#
#     def calc_grad(self):
#         ''' Stores the gradients, grad, in a tensor, where each row corresponds to each the
#             data from the Variable of the gradients '''
#         grad      = torch.autograd.grad([self.joint], self.params, grad_outputs= torch.ones(self.params[0].data))
#         gradients = torch.Tensor(len(self.params), self.params[0].data.size())
#         for i in range(len(self.params)):
#            gradients[i][:] = grad[i][0].data.unsqueeze(0) # ensures that each row of the grads represents a params grad
#         return gradients
#
#     def  calc_pot(self):
#         ''' Calculates the ' log potential function' needed to calculate the Hamiltonian '''
#         return self.joint.data



#
# class LogPotentialDisc():
#     ''' TO DO'''

class HMCsampler(object):
    '''
    Object - The potential energy function, that contains information regarding the joint.
    Notes:  the params from FOPPL graph will have to all be passed to - maybe
    Methods
    -------
    leapfrog_step - preforms the integrator step in HMC
    hamiltonian   - calculates the value of hamiltonian
    acceptance    - calculates the acceptance probability
    run_sampler

    Attributes
    ----------

    '''
    def __init__(self, joint, params, p, burn_in= 100, num_steps= 1000, M= None,  min_step= None, max_step= None,\
                 min_traj= None, max_traj= None):
        # TO DO: May have to pass through the FOPPL-Potential interactor class, rather than 'joint' and params
        # Maybe better to have a joint class?
        self.params    = params
        self.p         = p
        self.burn_in   = burn_in
        self.n_steps   = num_steps
        if min_step is None:
            min_step = torch.Tensor(1).uniform_(0.01, 0.07)
        if max_step is None:
            max_step = torch.Tensor(1).uniform_(0.14, 0.20)
        if max_traj is None:
            max_traj = torch.Tensor(1).uniform_(18, 25)
        if min_traj is None:
            min_traj = torch.Tensor(1).uniform_(5, 12)
        self.step_size = torch.Tensor(1).uniform_(min_step, max_step)
        self.traj_size = torch.Tensor(1).uniform_(min_traj, max_traj)
        self.kinetic   = KEnergy(M)
        self.potential = Program(joint, params)
        # TO DO : Implement a adaptive step size tuning from HMC
        # TO DO : Have a desired target acceptance ratio

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

        log_potential = self.log_potential
        step_size = self.step_size
        kinetic = self.kinetic
        n_steps = self.nstepsv

        # Start by updating the momentum a half-step
        p = p0 + 0.5 * step_size * self.potential.calc_grad()
        # Initalize x to be the first step
        x0 = x0 + step_size * self.kinetic.gauss_ke_grad(p)
        # If the gradients are not zeroed then they will blow up. This leads
        # to an exponential increase in kinetic and potential energy.
        # As the position and momentum increase unbounded.
        for i in range(n_steps - 1):
            # Compute gradient of the log-posterior with respect to x
            # Update momentum
            p = p + step_size * log_potential(x0,grad=True)

            # Update x
            x0.data = x0.data + step_size * kinetic(p,grad=True)
            x0.grad.data.zero_()

        # Do a final update of the momentum for a half step

        p = p + 0.5 * step_size * log_potential(x0,grad=True)
        xproposed = x0
        pproposed = p
        # return new proposal state
        return xproposed, pproposed

    def hamiltonian(self, x, p):
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
        U = self.log_potential(x)
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
        x, p = self.leapfrog_steps()
        orig = self.hamiltonian(self.x0, self.p0)
        current = self.hamiltonian(x, p)
        alpha = torch.min(torch.exp(orig - current))
        # calculate acceptance probability
        p_accept = min(1, alpha)
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
        n_samples = self.nsamples
        n_dim = self.ndim
        n_vars = self.nvars
        min_step = self.min_step
        max_step = self.max_step
        min_traj = self.min_traj
        max_traj = self.max_traj
        burn_in = self.burn_in
        samples = torch.Tensor(n_samples, n_dim)
        samples[0] = self.x0.data
        for i in range(n_samples - 1):
            temp = self.acceptance()
            # update the intial value of self.x0 globally
            self.x0 = temp
            samples[i] = temp.data
            # update parameters and draw new momentum
            self.step_size = np.random.uniform(min_step, max_step)
            self.n_steps = int(np.random.uniform(min_traj, max_traj))
            self.p0 = torch.randn(n_vars, n_dim)

        target_acceptance = self.count / (n_samples - 1)
        sampl1np = samples[burn_in:, :].numpy()
        #    print(sampl1np)
        sam1mean = sampl1np.mean(axis=0)
        samp1_var = np.cov(sampl1np.T)
        print('****** TRUE MEAN/ COV ******')
        print('True mean: ', np.zeros((1, n_dim)))
        print('True cov: ', self.cov)
        print()
        print('****** EMPIRICAL MEAN/COV USING HMC ******')
        print('empirical mean : ', sam1mean)
        print('empirical_cov  :\n', samp1_var)
        print('Average acceptance rate is: ', target_acceptance)

class Statistics(object):
    '''A class that contains .mean() and .var() methods and returns the sampled
    statistics given an MCMC chain. '''
# return samples[burn_in:, :], target_acceptance

# class Distribution_creator(object):
#    '''Pass in a string for the desired distrbution required
#    and returns a subset of the scipy.stats object for the given distrubution.
#    Each object will have a log_pdf , as required for the potential.
#
#    Parameters
#    ----------
#    distribution - str
#
#    Output
#    ------
#    distribution object
#    '''
#    def __init__(self, distribution):
#        self.distribution = distribution


def main():
    n_dim = 5
    n_samples = 100
    burnin = 0
    n_vars = 1
    minstep = 0.03
    maxstep = 0.18
    mintraj = 5
    maxtraj = 15
    # Intialise both trajectory length and step size
    step_size = np.random.uniform(minstep, maxstep)
    n_steps = int(np.random.uniform(mintraj, maxtraj))
    xinit = Variable(torch.randn(n_vars, n_dim), requires_grad=True)
    pinit = torch.randn(n_vars, n_dim)
    hmc_sampler = HMCsampler(x0=xinit,
                             p0=pinit,
                             ndim=n_dim,
                             nsamples=n_samples,
                             burn_in=burnin,
                             nvars=n_vars,
                             nsteps=n_steps,
                             stepsize=step_size,
                             min_step=minstep,
                             max_step=maxstep,
                             min_traj=mintraj,
                             max_traj=maxtraj,
                             count=0,
                             log_potential=log_potential_fn,
                             kinetic=kinetic_fn)
    hmc_sampler.run_sampler()


main()