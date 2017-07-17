#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:45:04 2017

@author: bradley

A simple implementation of Metropolis-Hasting algorithm 

"""
import numpy as np
import matplotlib.pyplot as plt

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
    n        = 20 # number of samples
    ybar     = 0.0675
    x_chain  = []
    x_current = np.random.randn(1)
    x_chain.append(x_current)
    for i in range(100):
        y_sample          = IS_likelihood(x_current,ybar,n)
        x_proposal       = IS_proposal(y_sample)
        x_new             = alpha(x_current,x_proposal,n,ybar)
        # This step is to highlight that the new x_new value, becomes our
        # new current x.  We could have just labeled x_new as x_current
        x_current         = x_new
        x_chain.append(x_new)
    return x_chain

def alpha(x_current,x_proposal,y_sample,n,ybar):
    '''Decides whether to accept or reject the proposal. As the proposal
    distribution is not dependent on x, we have a simple acceptance ration
    of just the priors. '''
    acceptance = IS_proposal(x_proposal)/IS_target(x_current,n,ybar)
    u = np.random.uniform(0,1)
    r = acceptance
    if r >= 1:
        return x_proposal
    # return x proposal if the sampled uniform value u, is less than the 
    # acceptance ratio of r. We accept x_proposal, with prob u. 
    if u < r:
        return x_proposal
    elif u >= r:
        return x_current
        
def IS_target(x, ybar,n):
    '''This is the posterior '''
    return IS_likelihood(x, ybar,n) * IS_prior(x)
           
def IS_proposal(y):
    return 1/(1 + y**2)

def IS_prior(x):
    return 1/ (1 + x**2)

def IS_likelihood(x,ybar,n):
    inside = 0.5*n*((x - ybar)**2)
    return np.exp(-inside)

def plot_histogram(chain):
     bins = np.linspace(math.ceil(min(chain)),math.floor(max(chain)),5)
     plt.figure(2)
     plt.xlim([min(chain)-1, max(chain)+1])
     
     plt.hist(chain,bins=bins, alpha = 0.7)
     plt.title('Histogram of markov chain')
     plt.xlabel('markov chain values')
     plt.ylabel('count')
     
def main():
    sampled_x = independence_sampler()
    plt.figure(1)
    no_samples = np.arange(0,len(sampled_x),1)
    plt.plot(no_samples,sampled_x)
    print(np.mean(sampled_x))
    #plot_histogram(sampled_x)
    
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
    