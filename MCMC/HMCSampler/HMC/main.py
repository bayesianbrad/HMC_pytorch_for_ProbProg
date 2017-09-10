#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:40
Date created:  06/09/2017

License: MIT
'''
from HMC.hmc_class import HMCsampler as HMC
from Utils.plotting_and_saving import Plotting
models = ['conjgauss', 'conditionalif', 'linearreg', 'hierarchial', 'mixture']
i = 3
# for i in range(len(models)):
#     hmcsampler  = HMC(burn_in=0, n_samples = 500, model= models[i])
#     samples, samples_with_burnin, mean =  hmcsampler.run_sampler()
#     plots = Plotting(samples,samples_with_burnin,mean, model = models[i])
#     plots.call_all_methods()
hmcsampler  = HMC(burn_in=0, n_samples = 1000, model= models[i])
samples_with_burnin,samples, mean =  hmcsampler.run_sampler()
plots = Plotting(samples,samples_with_burnin,mean, model = models[i])
plots.call_all_methods()