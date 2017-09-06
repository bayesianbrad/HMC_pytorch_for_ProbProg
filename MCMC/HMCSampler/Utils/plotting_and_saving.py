#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:41
Date created:  06/09/2017

License: MIT
'''
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable
from matplotlib import pyplot as plt

class Plotting():

    def __init__(self, samples,samples_with_burnin, mean, cov= None):
        if isinstance(samples, Variable):
            self.samples = samples.data.numpy()
        else:
            self.samples = samples.numpy()
        if isinstance(samples_with_burnin, Variable):
            self.samples_with_burnin = samples_with_burnin.data.numpy()
        else:
            self.samples_with_burnin  = samples_with_burnin.numpy()
        self.mean    = mean
        if cov is not None:
            self.cov = cov
        self.PATH  = '/Users/bradley/Documents/Aims_work/Miniproject2/Project_notes/MCMC/report_data_and_plots'

    def plot_trace(self):
        '''

        :param samples:  an nparray
        :param parameters:  Is a list of which parameters to take the traces of
        :return:
        '''
        print('Saving trace plots.....')
        fig, ax = plt.subplots()
        iter = np.arange(0, np.shape(self.samples)[0])
        ax.plot(iter, self.samples, label= ' values ')
        ax.set_title('Trace plot for the parameters after burn in')
        fname = 'figures/trace_uniform.png'
        fig.savefig(os.path.join(self.PATH, fname), dpi=400)

    def histogram(self):
        print('Saving histogram')
        weights = np.ones_like(self.samples) / float(len(self.samples))
        fig, ax = plt.subplots()
        ax.hist(self.samples,  bins = 'auto', normed=1)
        ax.set_xlabel(' Samples ')
        ax.set_ylabel('Density')
        ax.set_title('Histogram of samples \n' + r'$\mu_{\mathrm{emperical}}$' + r'$={}$'.format(self.mean[0]))
        # plt.axis([40, 160, 0, 0.03])
        ax.grid(True)
        fname = 'figures/histogram_uniform.png'
        fig.savefig(os.path.join(self.PATH,fname), dpi = 400)
        # plt.show()
    def save_data(self):
        print('Saving data....')
        df1 = pd.DataFrame(self.samples)
        df2 = pd.DataFrame(self.samples_with_burnin)
        path1 =  'data/samples_after_burnin.csv'
        print(path1)
        path2 =  'data/samples_with_burnin.csv'
        df1.to_csv(os.path.join(self.PATH,path1))
        df2.to_csv(os.path.join(self.PATH,path2))

    def call_all_methods(self):
        self.plot_trace()
        self.histogram()
        self.save_data()