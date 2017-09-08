#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:40
Date created:  06/09/2017

License: MIT
'''
from HMC.hmc_class_development import HMCsampler as HMC
from Utils.plotting_and_saving import Plotting
models = ['conjgauss', 'conditionalif', 'linearreg']
hmcsampler  = HMC(burn_in=0, n_samples = 2000, dim=1, model= models[0])
samples, samples_with_burnin, mean =  hmcsampler.run_sampler()
plots = Plotting(samples,samples_with_burnin,mean, model = models[0])
plots.call_all_methods()