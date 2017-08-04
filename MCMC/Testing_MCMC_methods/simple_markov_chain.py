#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:08:10 2017

@author: bradley

A simple markov chain, based on first order auto-regressive process with
lag 1 and autocorrelation 0.5 i.e N(0.5x(t), 1.0)

A markov chain P(X^(t+1) | X^(t)) is only dependent on the previous time and
sampled from the conditional. 


"""
import numpy as np
import matplotlib.pyplot as plt
import math

def simple_markov_chain(init):
    x_t     = init
    chain   = []
    
    for i in range(10000):
        x_t_next = np.random.normal(0.5*x_t,1)
        x_t      = x_t_next
        chain.append(x_t)
    return chain

def plot_histogram(chain):
     bins = np.linspace(math.ceil(min(chain)),math.floor(max(chain)),100)
     plt.xlim([min(chain)-1, max(chain)+1])
     
     plt.hist(chain,bins=bins, alpha = 0.7)
     plt.title('Histogram of markov chain')
     plt.xlabel('markov chain values')
     plt.ylabel('count')
     
def main():
    np.random.seed(2)
    chains = []
    for i in range(2):
         init   = 2.32
         chains.append(simple_markov_chain(init))
    # plots
    plt.figure(1)
    for chain in chains:
        x = np.arange(len(chain))
        plt.plot(x,chain)
    plt.figure(2)
    for chain in chains:
        plot_histogram(chain)
        print(np.std(chain))
        print(np.mean(chain))