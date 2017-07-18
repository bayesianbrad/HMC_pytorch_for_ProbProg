#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:45:04 2017

@author: bradley

A simple implementation of Metropolis-Hasting algorithm 

"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.stats
# Just metropolis, i.e. q(x|y) = q(y|x) 
# target dist is choosen to be exp(-x^(2) / 2)
# proposal dist is cossen to be N(y| x,0.25)
def target_dist(x):
    return np.exp(-0.5 * x**2)
def proposal_dist(x,y):
    temp  = 1 / (2* 0.5**2)
    return np.exp(-temp*(x-y)**2)

def metropolis():
    return 0

    
def independence_sampler():
    '''An independence sampler assumes that the proposal distribution q(y|x)
     is only dependent on y. i.e q(y|x) = q(y)
    The posterior is proportional to the likelihood * prior of the laten
    variables. That is: π0(X) = 1 / π(1 + \X^{2}) * L(y|x)
    where L(y|x) = exp(-(X - \bar{y})**2 / 2)
    Let the proposal q(y|x) = q(y)  = 1/π(1+y**2)
    '''
    x_chain  = []
    # Prior on the intial state x_0 is π^{0} ~ N(0,1)
    x_current = random.gauss(0,1)
    x_chain.append(x_current)
    for i in range(5000):
        x_proposal        = IS_proposal(x_current)
        x_new             = alpha(x_current,x_proposal)
        # This step is to highlight that the new x_new value, becomes our
        # new current x.  We could have just labeled x_new as x_current
        x_current         = x_new
        x_chain.append(x_new)
    return x_chain

def alpha(x_current,x_proposal):
    '''Decides whether to accept or reject the proposal. As the proposal
    distribution is not dependent on x, we have a simple acceptance ration
    of just the priors. '''
    acceptance = IS_target(x_proposal)/IS_target(x_current)
    u = np.random.uniform(0,1)
    r = acceptance

    # return x proposal if the sampled uniform value u, is less than the 
    # acceptance ratio of r. We accept x_proposal, with prob u. 
    if u < r:
        return x_proposal
    elif u >= r:
        return x_current
        
def IS_target(x):
    '''This is the posterior "target distribution - in this instance" '''
    return 1/ (1 + x**2)
           
def IS_proposal(x_current):
    '''q(x^{t+1} | x^{t}) ~ N(x^{t},1)'''
    return np.random.normal(x_current,1)



def plot_histogram(samples):
     bins = np.linspace(math.ceil(min(samples)),math.floor(max(samples)),1000)
     plt.figure(2)
     plt.xlim([min(samples)-5, max(samples)+5])
     histogram_data = np.histogram(samples, bins = 'auto', density= False)
     # 'auto' automatically finds optimum number of bins
     plt.hist(samples,bins= 'auto',normed = True, alpha =0.5, label = 'Histogram of samples')
     plt.title('Histogram of proposed x-values')
     plt.xlabel(r'samples,$ x $')
     plt.ylabel(r"$p(x)$, target distribtution")
     plot_t_dist()

def plot_t_dist():
    x          = np.linspace(-6,6,10000)
    df         = 1
    t_dist_pdf = scipy.stats.t.pdf(x,df)
    plt.plot(x, t_dist_pdf,label = r't-dist $\nu = 0$ ')
    plt.legend()
    mean, var = t_dist_pdf.stats(df, moments = 'mv')

def main():   
    random.seed(1)
    sampled_x = independence_sampler()
    plt.figure(1)
    no_samples = np.arange(0,len(sampled_x),1)
    plt.plot(no_samples,sampled_x)
    unit_sample = sampled_x / np.linalg.norm(sampled_x)
    print(np.mean(unit_sample))
    plt.figure(2)
    plot_histogram(sampled_x)
#    np.random.seed(1)
#    x = np.linspace(-3,3,1000)
#    y = np.random.randn(1000)
#    plt.figure(1)
#    proposal = proposal_dist(x,y[2])
#    target   = target_dist(x)
# The target will remain the same, but the proposal depends on where you
# are. 
#    
#    plt.plot(x,proposal,'r-')
#    plt.plot(x,target,'b.')
#    proposal = proposal_dist(x,y[423])
#    plt.plot(x,proposal,'k--')
    