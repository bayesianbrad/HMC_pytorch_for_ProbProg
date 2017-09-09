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
import errno
from torch.autograd import Variable
from matplotlib import pyplot as plt

class Plotting():

    def __init__(self, samples,samples_with_burnin, mean, model = None, cov= None):
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
        if model is not None:
            # model      = model + 'small'
            model = model
            self.PATH  = '/Users/bradley/Documents/Aims_work/Miniproject2/Project_notes/MCMC/report_data_and_plots' +'/' + model
            os.makedirs(self.PATH, exist_ok=True)
            self.PATH_fig = os.path.join(self.PATH, 'figures')
            os.makedirs(self.PATH_fig, exist_ok=True)
            self.PATH_data =  os.path.join(self.PATH, 'data')
            os.makedirs(self.PATH_data, exist_ok=True)
        else:
            self.PATH = '/Users/bradley/Documents/Aims_work/Miniproject2/Project_notes/MCMC/report_data_and_plots'
    def plot_trace(self):
        '''

        :param samples:  an nparray
        :param parameters:  Is a list of which parameters to take the traces of
        :return:
        '''
        print('Saving trace plots.....')
        fig, ax = plt.subplots()
        iter = np.arange(0, np.shape(self.samples_with_burnin)[0])
        if np.shape(self.samples)[1] > 1:
            for i in range(np.shape(self.samples)[1]):
                ax.plot(iter, self.samples[:,i], label='Parameter {0} '.format(i))
                ax.set_title('Trace plot for the parameters')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Sampled values of the Parameter')
                plt.legend()
                fname = 'trace.png'
                fig.savefig(os.path.join(self.PATH_fig, fname), dpi=400)
        else:
            ax.plot(iter, self.samples, label= 'Parameter')
            ax.set_title('Trace plot for the parameter')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Sampled values of the Parameter')
            plt.legend()
            fname = 'trace.png'
            fig.savefig(os.path.join(self.PATH_fig, fname), dpi=400)

    def histogram(self):
        print('Saving histogram.....')
        weights = np.ones_like(self.samples) / float(len(self.samples))
        fig, ax = plt.subplots()
        if np.shape(self.samples)[1] > 1:
            for i in range(np.shape(self.samples)[1]):
                ax.hist(self.samples[:,i],  bins = 'auto', normed=1, label= r'$\mu_{\mathrm{emperical}}$' + '=' + '{0}'.format(
                        self.mean.data[0][i]))
                ax.set_title('Histogram of samples ')
                ax.set_xlabel(' Samples ')
                ax.set_ylabel('Density')
                ax.grid(True)
            plt.legend()
            fname = 'histogram.png'
                # Ensures directory for this figure exists for model, if not creates it
            fig.savefig(os.path.join(self.PATH_fig, fname), dpi=400)


        else:
            ax.hist(self.samples, bins='auto', normed=1, label= r'$\mu_{\mathrm{emperical}}$' + '=' + '{0}'.format(self.mean.data[0][0]))
            ax.set_title(
                'Histogram of samples')
            ax.set_xlabel(' Samples ')
            ax.set_ylabel('Density')
        # plt.axis([40, 160, 0, 0.03])
            ax.grid(True)
            plt.legend()
        # Ensures directory for this figure exists for model, if not creates it
            fig.savefig(os.path.join(self.PATH_fig,'histogram.png' ), dpi = 400)
        # plt.show()

    def save_data(self):
        print('Saving data....')
        df1 = pd.DataFrame(self.samples)
        df2 = pd.DataFrame(self.samples_with_burnin)
        # Ensures directory for this data exists for model, if not creates it
        path1 =  'samples_after_burnin.csv'
        path2 =  'samples_with_burnin.csv'
        df1.to_csv(os.path.join(self.PATH_data,path1))
        df2.to_csv(os.path.join(self.PATH_data,path2))

    def call_all_methods(self):
        self.plot_trace()
        self.histogram()
        self.save_data()