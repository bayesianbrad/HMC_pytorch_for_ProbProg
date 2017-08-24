#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:14:09 2017

@author: bradley
"""

import torch
from torch.autograd import Variable
torch.manual_seed(1234)
x = Variable(torch.Tensor(2,2).uniform_(), requires_grad = True)
print(x)
y = 2*x
y = torch.sum(y)
y.backward()
p = x.grad.data
print(p)
x.grad.data.zero_()
z  =2*x**2
z = torch.sum(z)
z.backward()
q = x.grad.data.clone()
print(q)
print(p - q)