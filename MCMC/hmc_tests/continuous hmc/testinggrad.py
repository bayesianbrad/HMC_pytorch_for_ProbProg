#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:18:57 2017

@author: bradley
"""
import numpy as np
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

d = 2
n = 1

p = Variable(torch.randn(n,d).type(dtype), requires_grad = True)

# Construct diagonal matrix in torch
M_inv = np.random.randn(2,2)
M_inv = np.diag(np.asarray(M_inv, dtype = np.float32)[:,0])
M_inv = torch.from_numpy(M_inv)

M_inv = Variable(M_inv,  requires_grad = False)

Kinetic = p.mm(M_inv).mm(torch.transpose(p,0,1))

Kinetic.backward()  
# Manually zero the gradients 
dk_dp = p.grad.data


# Testing 
print(dk_dp)
print('\n', 2*p.mm(M_inv))
print('\n', p) 
print('\n', M_inv)
p.grad.data.zero_()
print(p.grad)
