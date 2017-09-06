#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:31:26 2017

@author: bradley
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from torch.autograd import Variable
from Utils.core import VariableCast
from Utils.program import program_simple as program
from Utils.kinetic import Kinetic

# np.random.seed(1234)
# torch.manual_seed(1234)



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
            self.min_step = np.random.uniform(0.01, 0.07)
        else:
            self.min_step = min_step
        if max_step is None:
            self.max_step = np.random.uniform(0.07, 0.18)
        else:
            self.max_step = max_step
        if max_traj is None:
            self.max_traj = np.random.uniform(18, 25)
        else:
            self.max_traj = max_traj
        if min_traj is None:
            self.min_traj = np.random.uniform(0, 18)
        else:
            self.min_traj = min_traj
        self.M         = M
        self.step_size = np.random.uniform(self.min_step, self.max_step)
        self.traj_size = int(np.random.uniform(self.min_traj, self.max_traj))
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
        p      = p_init + 0.5 * step_size * grad_init
        values = values_init + step_size * self.kinetic.gauss_ke(p, grad= True)
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
        return -logjoint + T

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

        # generate proposals
        values, p    = self.leapfrog_steps(p_init, values_init, grad_init)

        # calculate new hamiltonian given current
        logjoint_prop, _ = self.potential.eval(values, grad= False)

        current      = self.hamiltonian(logjoint_prop, p)
        alpha        = torch.min(torch.exp(orig - current))
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

    def plot_trace(self, samples):
        '''

        :param samples:  an nparray
        :param parameters:  Is a list of which parameters to take the traces of
        :return:
        '''
        print('This plotting traces.....')
        fig, ax = plt.subplots()
        iter = np.arange(0, np.shape(samples)[0])
        ax.plot(iter, samples, label= ' values ')
        ax.set_title('Trace plot for the parameters')
        fname = '../../report_figures/trace_uniform.png'
        plt.savefig(fname, dpi=400)

    def histogram(self, samples, mean):
        weights = np.ones_like(samples) / float(len(samples))
        plt.clf()
        plt.hist(samples,  bins = 'auto', normed=1)
        plt.xlabel(' Samples ')
        plt.ylabel('Density')
        plt.title('Histogram of samples \n' + r'$\mu_{\mathrm{emperical}}$' + r'$={}$'.format(mean[0]))
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        fname = '../../report_figures/histogram_uniform.png'
        plt.savefig(fname, dpi = 400)
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
        # if isinstance(dict, values_init):
        #     print('do something else')
        # else:
        #     logjoint_init, values_init, grad_init = self.potential.generate()
        temp = self.acceptance(logjoint_init, values_init, grad_init)
        n_dim = values_init.size()[1]
        samples      = Variable(torch.zeros(self.n_samples,n_dim))
        samples[0, :] = temp.data
        self.step_size = np.random.uniform(self.min_step, self.max_step)
        self.n_steps = int(np.random.uniform(self.min_traj, self.max_traj))
        # Then run for loop from 2:n_samples
        samples      = Variable(torch.zeros(self.n_samples,1))

        for i in range(self.n_samples-1):
            # TO DO: Get this bit sorted
            # print(' Iteration ', i)
            # print(' Samples ' , temp.data)
            # print(' Samples type ', type(temp))
            logjoint_init, grad_init = self.potential.eval(temp, grad2= True)
            temp = self.acceptance(logjoint_init,temp, grad_init)
            # store accepted sample
            # print(' Temp data {0} interation {1} '.format(temp.data, i))
            # print(temp)
            samples[i+1,:] = temp.data
            # update parameters and draw new momentum
            self.step_size = np.random.uniform(self.min_step, self.max_step)
            self.traj_size = int(np.random.uniform(self.min_traj, self.max_traj))

        target_acceptance = self.count / (self.n_samples)
        samples_reduced   = samples[self.burn_in:, :]
        mean = torch.mean(samples_reduced,dim=0, keepdim= True)
        print(samples_reduced)

        print()
        print('****** EMPIRICAL MEAN/COV USING HMC ******')
        print('empirical mean : ', mean)
        print('Average acceptance rate is: ', target_acceptance)
        self.plot_trace(samples.data.numpy())
        self.histogram(samples_reduced.data.numpy(), mean.data.numpy())

# class Statistics(object):
#     '''A class that contains .mean() and .var() methods and returns the sampled
#     statistics given an MCMC chain. '''
# # return samples[burn_in:, :], target_acceptance


class Plotting():

    def __init__(self, samples, mean, cov= None):
        if isinstance(Variable, samples):
            self.samples = samples.data.numpy()
        else:
            self.samples = samples.numpy()
        self.mean    = mean
        if cov is not None:
            self.cov = cov

    def plot_trace(self, samples, parameters):
        '''

        :param samples:  an nparray
        :param parameters:  Is a list of which parameters to take the traces of
        :return:
        '''
        print('This plotting traces.....')
        fig, ax = plt.subplots()
        iter = np.arange(0, np.shape(samples)[0])
        ax.plot(iter, samples, label= ' values ')
        ax.set_title('Trace plot for the parameters')
        plt.show()

    def histogram(self, samples, mean):
        n, bins, patches = plt.hist(samples, normed=1, facecolor='green', alpha=0.75)

        plt.xlabel(' Samples ')
        plt.ylabel('Probability')
        plt.title('Histogram of samples' + 'r $\mu={}$'.format([0]))
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(True)

        plt.show()
def main():
    # n_dim = 5
    # n_samples = 10000
    # burnin = 0
    # n_vars = 1
    # minstep = 0.03
    # maxstep = 0.18
    # mintraj = 5
    # maxtraj = 15
    hmcsampler  = HMCsampler(burn_in=100, n_samples= 1000)
    hmcsampler.run_sampler()


main()