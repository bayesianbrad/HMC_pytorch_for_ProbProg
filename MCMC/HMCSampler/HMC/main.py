#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:40
Date created:  06/09/2017

License: MIT
'''
from HMC.hmc_class_general import HMCsampler as HMC
from Utils.plotting_and_saving import Plotting
hmcsampler  = HMC(burn_in=100, n_samples= 200)
samples, samples_with_burnin, mean =  hmcsampler.run_sampler()
plots = Plotting(samples,samples_with_burnin,mean)
plots.call_all_methods()
