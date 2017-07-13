#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:19:11 2017

@author: bradley
"""

from __future__ import print_function
import torch
import numpy as np

# Construct a 5 x 3 matrix

def intialising_matrices_basic_operations():
    x = torch.Tensor(5, 3)
    #print(x)
    
    # To construct a randomly intialised matrix
    
    x = torch.rand(5, 3)
    #print(x)
    
    # Get its size, this returns a tuple
    
    print(x.size())
    
    y = x.size()
    #print(y)
    #print(y[0])
    
    ## Operations
    
    # Addition 1
    y = torch.rand(5,3)
    #print(x + y)
    # Addition 2
    #print(torch.add(x, y))
    
    # Saving the output of an operation, for
    # example, addition
    
    result = torch.Tensor(5,3)
    torch.add(x, y, out = result)
    print(result)
    
    # Addition in place
    # Adds x to y
    
    y.add_(x)
    print(y)
    
    # Any operation that mutates a tensor in place is  post-fixed with an _
    # i.e
    x.copy_(y)  # will change x
    
def numpy_bridge():
    # We can transform a tensor to an nparray and an nparray to a tensor
    a = torch.ones(5)
    print(a)
    
    b = a.numpy()
#    print(b)
#    print(type(b), '\n', print(type(a)))
    
    a.add_(1)
    print(a)
    print(b)
    
    
    ## NP array ---> tensor
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out = a)
    print(a)
    print(b)
    
    
    # Tensors can be moved onto GPU using the .cuda function 
    # let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)