#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:47
Date created:  06/09/2017

License: MIT
'''
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
from HMC import integrator

np.random.seed(1234)
torch.manual_seed(1234)


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
            self.min_traj = np.random.uniform(1, 18)
        else:
            self.min_traj = min_traj
        self.M         = M
        self.step_size = np.random.uniform(self.min_step, self.max_step)
        self.traj_size = int(np.random.uniform(self.min_traj, self.max_traj))
        self.potential = program()
        self.integrator= integrator(self.step_size, self.traj_size)
        # to calculate acceptance probability
        self.count     = 0

        # TO DO : Implement a adaptive step size tuning from HMC
        # TO DO : Have a desired target acceptance ratio
        # TO DO : Implement a adaptive trajectory size from HMC
    def sample_momentum(self,values,dim):
        assert(isinstance(values, list))
        return VariableCast(torch.randn(len(values),dim))

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

        Output
        ------
        returns accepted or rejected proposal
        '''

        # generate initial momentum

        #### FLAG
        dim    = values_init[0].data.size()[0]
        p_init = self.sample_momentum(values_init,dim)
        # generate kinetic energy object.
        self.kinetic = Kinetic(p_init,self.M)
        # calc hamiltonian  on initial state
        orig         = self.hamiltonian(logjoint_init, p_init)

        # generate proposals
        values, p    = self.integrator.leapfrog(p_init, values_init, grad_init)

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
        print(' The sampler is now running')
        logjoint_init, values_init, grad_init = self.potential.generate()
        temp = self.acceptance(logjoint_init, values_init, grad_init)
        n_dim = values_init.size()[1]
        samples      = Variable(torch.zeros(self.n_samples,n_dim))
        samples[0, :] = temp.data
        self.step_size = np.random.uniform(self.min_step, self.max_step)
        self.n_steps = int(np.random.uniform(self.min_traj, self.max_traj))
        # Then run for loop from 2:n_samples
        samples      = Variable(torch.zeros(self.n_samples,1))

        for i in range(self.n_samples-1):
            logjoint_init, grad_init = self.potential.eval(temp, grad2= True)
            temp = self.acceptance(logjoint_init,temp, grad_init)
            samples[i+1,:] = temp.data
            # update parameters and draw new momentum
            self.step_size = np.random.uniform(self.min_step, self.max_step)
            self.traj_size = int(np.random.uniform(self.min_traj, self.max_traj))

        target_acceptance = self.count / (self.n_samples)
        samples_reduced   = samples[self.burn_in:, :]
        mean = torch.mean(samples_reduced,dim=0, keepdim= True)
        print()
        print('****** EMPIRICAL MEAN/COV USING HMC ******')
        print('empirical mean : ', mean)
        print('Average acceptance rate is: ', target_acceptance)

        return samples_reduced, samples, mean
