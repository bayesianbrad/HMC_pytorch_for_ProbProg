#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:13:45 2017

@author: bradley
"""

import torch
from  torch.autograd import Variable
from  torch.utils.data import TensorDataset, DataLoader

# Define new autogram function

class ReLU(torch.autograd.Function):
    #forward pass
    def forward(self, x):
        self.save_for_backward(x)
        return x.clamp(min = 0)
    #backward pass
    def backward(self, grad_y):
        x,               = self.saved_tensors
        grad_input      = grad_y.clone()
        grad_input[x<0] = 0
        return grad_input

dtype             = torch.FloatTensor
N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

w1 = Variable(torch.randn(D_in, H), requires_grad = True)
w2 = Variable(torch.randn(H, D_out), requires_grad = True)

loader  = DataLoader(TensorDataset(x,y), batch_size = 8)
learning_rate = 1e-6

def network1():
    for t in range(500):
        y_pred  =x.mm(w1).clamp(min=0).mm(w2)
        loss    = (y_pred - y).pow(2).sum()
        
        try:
            if w1.grad: w1.grad.data.zero_()
            if w2.grad: w2.grad.data.zero_()
            loss.backward()
        except  RuntimeError:
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            loss.backward()
        
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

def network2():
    for t in range(500):
        relu    = ReLU()
        y_pred  = relu(x.mm(w1)).mm(w2)
        loss    = (y_pred - y).pow(2).sum()
        
        try:
            if w1.grad: w1.grad.data.zero_()
            if w2.grad: w2.grad.data.zero_()
            loss.backward()
        except  RuntimeError:
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            loss.backward()
        
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
network2()