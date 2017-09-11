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
i = 1
for i in range(len(models)-2):
    hmcsampler  = HMC(burn_in=0, n_samples = 5000, model= models[i])
    samples, samples_with_burnin, mean =  hmcsampler.run_sampler()
    plots = Plotting(samples,samples_with_burnin,mean, model = models[i])
    plots.call_all_methods()
# hmcsampler  = HMC(burn_in=0, n_samples = 5000, model= models[i])
# samples_with_burnin,samples, mean =  hmcsampler.run_sampler()
# plots = Plotting(samples,samples_with_burnin,mean, model = models[i])
# plots.call_all_methods()