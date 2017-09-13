#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:04:34 2017

@author: bradley
"""

#N(x;0,1)N(1;1,1)^I(x>0)N(1;-1,1)^I(x<0).

import numpy as np
import scipy.stats as ss
import matplotlib as mpl
mpl.use('pgf')
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics import tsaplots

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": [4,4],     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
second = np.loadtxt('samples_after_burnin.csv', delimiter=',', usecols=(1,1), unpack=True)
data  =second[1,:].flatten()
px = ss.norm(loc=0, scale=1)
x = np.random.randn(100000)
px = px.pdf(x)
py1        = ss.norm(loc=1, scale =1)
py2       = ss.norm(loc=-1,scale=1)
y  = np.ones(np.shape(x))
py1 =py1.pdf(y)
py2 =py2.pdf(y)  

mean = np.mean(data)
print(mean)
posterior  =  px*py1**(x>0)*py2**(x<0)
fig,ax = plt.subplots(1,1)
ax.plot(x, posterior, 'r.', label='Unnormalized density')

ax.hist(data, bins='auto', normed=1, label= r'$\mu_{\mathrm{emperical}}$' + '=' + '{0}'.format(mean))
ax.set_title(
    'Histogram of samples')
ax.set_xlabel(' Samples ')
ax.set_ylabel('Density')
# plt.axis([40, 160, 0, 0.03])
plt.legend()
# Ensures directory for this figure exists for model, if not creates it
fig.savefig('histogram_with_density.pdf')
# plt.show()
print('Finished')