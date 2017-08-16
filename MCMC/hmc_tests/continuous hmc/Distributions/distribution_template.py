#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:55:03 2017

@author: bradley
"""

# Random variables interface:

class RandomVariable:
    def size(self)        # --> (batch_size, rv_dimension)
        
    def log_pdf(self, x)  # --> [batch_size]

    def sample(self)      # --> [batch_size, rv_dimension]

    def entropy(self)     # --> [batch_size]


# Implemented random variables:

Normal(size=(batch_size, rv_dimension), cuda=cuda)
Normal(mu, sd)

Categorical(size=(batch_size, rv_dimension), cuda=cuda)
Categorical(p)

Bernoulli(size=(batch_size, rv_dimension), cuda=cuda)
Bernoulli(p)

Uniform(size=(batch_size, rv_dimension), cuda=cuda)

# KL-Divergence:

def kld(rv_from, rv_to)  # --> [batch_size]