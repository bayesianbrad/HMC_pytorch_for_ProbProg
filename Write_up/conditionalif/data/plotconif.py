#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:04:34 2017

@author: bradley
"""

#N(x;0,1)N(1;1,1)^I(x>0)N(1;-1,1)^I(x<0).

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

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


posterior  =  px*py1**(x>0)*py2**(x<0) / np.sqrt(2*np.pi*(py1**(x>0) + py2**(x<0)))
fig,ax = plt.subplots(1,1)
ax.plot(x, posterior, 'r.')
ax.hist(data,  bins = 'auto', normed=1)
plt.show()

