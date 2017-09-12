import torch
import numpy as np
import importlib
from torch.autograd import Variable
from Utils.core import VariableCast
from Utils.kinetic import Kinetic
from Utils.integrator import Integrator
from Utils.metropolis_step import Metropolis

class HMCsampler():

  def __init__(self, burn_in= 100, \
	n_samples= 1000, model = 'conjgauss',\
	M= None,  min_step= None,\ 
	max_step= None, min_traj= None,\ 
	max_traj= None):
    self.burn_in    = burn_in
    self.n_samples  = n_samples
    self.M          = M
    self.model      = model
    # External dependencies
    program         = \
    getattr(importlib.\
     import_module('Utils.program'), model)
    self.potential  = program()
    self.integrator= Integrator(\
    	self.potential,min_step,\
    	 max_step, min_traj, max_traj)


  def run_sampler(self):
    print(' The sampler is now running')
    logjoint_init, values_init, grad_init,\
     dim = self.potential.generate()
    metropolis = Metropolis(\
     self.potential,self.integrator, self.M)
    temp,count = metropolis.acceptance(\
     values_init,logjoint_init, grad_init)
    samples =\
     Variable(torch.zeros(self.n_samples,dim))
    samples[0]= temp.data.t()


    # Then run for loop from 2:n_samples
    for i in range(self.n_samples-1):
        logjoint_init, grad_init = \
        self.potential.eval(temp,grad_loop= True)
        temp, count = metropolis.acceptance(temp,\
         logjoint_init,grad_init)
        samples[i + 1, :] = temp.data.t()

    # Basic summary statistics
    target_acceptance =  count / (self.n_samples)
    samples_reduced   = samples[self.burn_in:, :]
    mean = torch.mean(samples_reduced,dim=0,\
     keepdim= True)

    return samples_reduced, samples, mean
