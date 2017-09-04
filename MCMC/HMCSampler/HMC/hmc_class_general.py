#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:31:26 2017

@author: bradley
"""

import torch
import numpy as np
from torch.autograd import Variable
import time
from Utils.core import VariableCast
from Utils.program import program
from Utils.kinetic import Kinetic
import math

np.random.seed(1234)
torch.manual_seed(1234)



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

class HMCsampler():
    '''
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
    def __init__(self, burn_in= 100, n_samples= 1000, M= None,  min_step= None, max_step= None,\
                 min_traj= None, max_traj= None):
        self.burn_in    = burn_in
        self.n_samples  = n_samples
        if min_step is None:
            self.min_step = torch.Tensor(1).uniform_(0.01, 0.07)
        else:
            self.min_step = min_step
        if max_step is None:
            self.max_step = torch.Tensor(1).uniform_(0.14, 0.20)
        else:
            self.max_step = max_step
        if max_traj is None:
            self.max_traj = torch.Tensor(1).uniform_(18, 25)
        else:
            self.max_traj = max_traj
        if min_traj is None:
            self.min_traj = torch.Tensor(1).uniform_(5, 12)
        else:
            self.min_traj = min_traj
        self.M         = M
        self.step_size = torch.Tensor(1).uniform_(self.min_step[0], self.max_step[0])[0]
        self.traj_size = int(torch.Tensor(1).uniform_(self.min_traj[0], self.max_traj[0])[0])
        self.potential = program()
        # to calculate acceptance probability
        self.count     = 0

        # TO DO : Implement a adaptive step size tuning from HMC
        # TO DO : Have a desired target acceptance ratio
        # TO DO : Implement a adaptive trajectory size from HMC
    def sample_momentum(self,values):
        return VariableCast(torch.randn(1,VariableCast(values).data.size()[0]))

    def leapfrog_steps(self, p_init, values_init, grad_init):
        '''Performs the leapfrog steps of the HMC for the specified trajectory
        length, given by num_steps
        Parameters
        ----------
            values_init
            p_init
            logjoint_init
            grad_init     - Description: contains the initial gradients of the joint w.r.t parameters.

        Outputs
        -------
            values -    Description: proposed new values
            p      -    Description: proposed new auxillary momentum
        '''

        step_size = self.step_size
        n_steps   = self.traj_size

        # Start by updating the momentum a half-step and values by a full step
        p = p_init + 0.5 * step_size * grad_init
        print(p)
        values = values_init + step_size * self.kinetic.gauss_ke(p, grad= True)
        print(values)
        for i in range(n_steps - 1):
            # range equiv to [2:nsteps] as we have already performed the first step
            # update momentum
            p = p + step_size * self.potential.eval(values, grad=True)
            # update values
            values = values_init + step_size * self.kinetic.gauss_ke(p, grad= True)


        # Do a final update of the momentum for a half step
        p = p + 0.5 * step_size * self.potential.eval(values, grad= True)
        # return new proposal state
        return values, p

    def hamiltonian(self, logjoint, p):
        """Computes the Hamiltonian  given the current postion and momentum
        H = U(x) + K(p)
        U is the potential energy and is = -log_posterior(x)
        Parameters
        ----------
        logjoint    - Type:torch.autograd.Variable \mathbb{R}^{1 \times 1}
        p           - Type:torch.Tensor \mathbb{R}^{1 \times D}.
                    Description: Auxiliary momentum
        log_potential :Function from state to position to 'energy'= -log_posterior

        Returns
        -------
        hamitonian : float
        """
        T = self.kinetic.gauss_ke(p, grad=False)
        # print('Debug HAM *******')
        # print(T)
        # print(logjoint)
        # print(type(logjoint), type(p))
        return logjoint + T

    def acceptance(self, logjoint_init, values_init, grad_init):
        '''Returns the new accepted state

        Parameters
        ----------
        x = xproposed
        x0
        p = pproposed
        p0

        Output
        ------
        returns accepted or rejected proposal
        '''

        # generate initial momentum
        p_init = self.sample_momentum(values_init)
        # generate kinetic energy object.
        self.kinetic = Kinetic(p_init,self.M)
        # calc hamiltonian  on initial state
        orig         = self.hamiltonian(logjoint_init, p_init)
        # print(' Init values ******')
        # print('momentum', p_init)
        # print('values', values_init)
        # print('logjoint', logjoint_init)
        # print('grad_init', grad_init)

        # generate proposals
        values, p    = self.leapfrog_steps(p_init, values_init, grad_init)
        # print(' Proposed values ********')
        # print(' momentum ', p)
        # print(' values ', values)

        # calculate new hamiltonian given current
        logjoint_prop, _ = self.potential.eval(values, grad= False)
        # print('logjoint prposed')
        # print(logjoint_prop)

        current      = self.hamiltonian(logjoint_prop, p)
        alpha = torch.min(torch.exp(orig - current))
        # print('alpha :', alpha)
        # calculate acceptance probability
        if isinstance(alpha, Variable):
            p_accept = torch.min(torch.ones(1,1), alpha.data)
        else:
            p_accept = torch.min(torch.ones(1,1),  alpha)
        # print(p_accept)
        if p_accept[0][0] > torch.Tensor(1,1).uniform_()[0][0]: #[0][0] dirty code to get integersr
            # Updates count globally for target acceptance rate
            self.count = self.count + 1
            return values
        else:
            return values_init

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
        logjoint_init, values_init, grad_init = self.potential.generate()
        if isinstance(dict, values_init):
            print('do something else')
        else:
            logjoint_init, values_init, grad_init = self.potential.generate()
        temp = self.acceptance(logjoint_init, values_init, grad_init)
        n_dim = values_init.size()[1]
        samples      = Variable(torch.zeros(self.n_samples,n_dim))
        samples[i, :] = temp.data
        self.step_size = np.random.uniform(self.min_step, self.max_step)
        self.n_steps = int(np.random.uniform(self.min_traj, self.max_traj))
        # Then run for loop from 2:n_samples
        samples      = Variable(torch.zeros(self.n_samples+1,1))

        for i in range(self.n_samples-1):
            # TO DO: Get this bit sorted
            print(' Iteration ', i)
            print(' Samples ' , temp.data)
            print(' Samples type ', type(temp))
            logjoint_init, grad_init = self.potential.eval(temp, grad2= True)
            temp = self.acceptance(logjoint_init,temp, grad_init)
            # store accepted sample
            print(' Temp data {0} interation {1} '.format(temp.data, i))
            samples[i+1,:] = temp.data
            # update parameters and draw new momentum
            self.step_size = torch.Tensor(1).uniform_(self.min_step[0], self.max_step[0])[0]
            self.traj_size = int(torch.Tensor(1).uniform_(self.min_traj[0], self.max_traj[0])[0])

        target_acceptance = self.count / (self.n_samples)
        # samples_reduced   = samples[self.burn_in:, :]
        # mean = torch.mean(samples,dim=0, keepdim= True)
        # print()
        # print('****** EMPIRICAL MEAN/COV USING HMC ******')
        # print('empirical mean : ', mean)
        # print('Average acceptance rate is: ', target_acceptance)

# class Statistics(object):
#     '''A class that contains .mean() and .var() methods and returns the sampled
#     statistics given an MCMC chain. '''
# # return samples[burn_in:, :], target_acceptance


def main():
    n_dim = 5
    n_samples = 100
    burnin = 0
    n_vars = 1
    minstep = 0.03
    maxstep = 0.18
    mintraj = 5
    maxtraj = 15
    hmcsampler  = HMCsampler(burn_in=0, n_samples= 100)
    hmcsampler.run_sampler()
    # Intialise both trajectory length and step size
    # hmc_sampler = HMCsampler(x0=xinit,
    #                          p0=pinit,
    #                          ndim=n_dim,
    #                          nsamples=n_samples,
    #                          burn_in=burnin,
    #                          nvars=n_vars,
    #                          min_step=minstep,
    #                          max_step=maxstep,
    #                          min_traj=mintraj,
    #                          max_traj=maxtraj,
    #                          count=0)
    # hmc_sampler.run_sampler()


main()