#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:49
Date created:  06/09/2017

License: MIT
'''
import torch
from Utils.kinetic import Kinetic
from torch.autograd import Variable

class Integrator():

    def __init__(self, step_size = 0.03, traj_size = 10):
        self.step_size   = step_size
        self.traj_size   = traj_size

    def list_to_tensor(self, params):
        ''' Unpacks the parameters list tensors and converts it to list

        returns tensor of  num_rows = len(values) and num_cols  = 1
        problem:
            if there are col dimensions greater than 1, then this will not work'''
        assert(isinstance(params, list))
        temp = Variable(torch.Tensor(len(params)).unsqueeze(-1))
        for i in range(len(params)):
            temp[i,:] = params[i]
        return temp
    def leapfrog(self, p_init, values, grad_init):
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

        values_init = self.list_to_tensor(values)
        self.kinetic = Kinetic(p_init)
        # Start by updating the momentum a half-step and values by a full step
        p = p_init + 0.5 * self.step_size * grad_init
        values = values_init + self.step_size * self.kinetic.gauss_ke(p, grad=True)
        for i in range(self.traj_size - 1):
            # range equiv to [2:nsteps] as we have already performed the first step
            # update momentum
            p = p + self.step_size * self.potential.eval(values, grad=True)
            # update values
            values = values + self.step_size * self.kinetic.gauss_ke(p, grad=True)

        # Do a final update of the momentum for a half step
        p = p + 0.5 * self.step_size * self.potential.eval(values, grad=True)
        # return new proposal state
        return values, p