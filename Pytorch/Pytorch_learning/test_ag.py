#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:15
Date created:  08/09/2017

License: MIT
'''

import scipy.stats as ss
import torch
import numpy as np
from torch.autograd import Variable

d = 1
a  = Variable(torch.ones(1,d), requires_grad = True)
b = Variable(2*torch.ones(1,d), requires_grad  = True)
c = Variable(3*torch.ones(1,d), requires_grad  = True)
x = Variable(torch.FloatTensor(1,d).zero_())
params_dic = {}
params  = [a,b,c]
for i in range(d):
    params_dic["var{0}".format(i)] = params[i]
#print(params_dic)
# as the first element needs to be assigned to x
x = params[0]
# then concatenate the rest of params
for i in range(len(params)):
    if i==0:
        x  = params[i]
    else:
        x = torch.cat((x,params[i]), dim = 0)
print(x)
# mvn = ss.multivariate_normal.logpdf(a.data.numpy(), mean  = np.zeros(d), cov =  np.eye(d))
# mvn.backward()
# unpack x
# for i in range(len(params)):
#     y += torch.log(params[i].mm(params[i].t()))
print('Debug size values', x.size())
print('Debug size values1', x[0, :].size())
print('Debug size values2', x[1, :].size())
def log_normal(value):
    value = value.unsqueeze(-1).t()
    mean = Variable(torch.rand(d).unsqueeze(-1).t())
    std  = Variable(torch.rand(d).unsqueeze(-1).t())
    true_grad = -(value.data  - mean.data)/std.data**2
    print('True grad for :', true_grad)
    return (-0.5 *  torch.pow(value - mean, 2) / std**2) -  torch.log(std)
y1 = log_normal(x[0,:])
y2 = 2*x[1,:].unsqueeze(-1).t()
y3 = log_normal(x[2,:])
# print(y1,y2,y3)
y = y1+y2+y3
# print(y)
gradients = torch.autograd.grad(outputs=y,inputs=x, grad_outputs = torch.ones(x.size()), retain_graph = True)
print(gradients[0].data * 1/x.size()[0])
# y.backward()
# print(a.grad)
# print(b.grad)
# print(c.grad)