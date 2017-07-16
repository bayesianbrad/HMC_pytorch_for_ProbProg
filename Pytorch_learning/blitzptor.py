#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:19:11 2017

@author: bradley
"""

from __future__ import print_function
import torch
import numpy as np
from torch import autograd
import torch.nn as nn
import torch.optim as optim

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
#==============================================================================
# Automatic differentiation       
#==============================================================================
def Auto_diff():
    '''Variable Attributes:
    data: Wrapped tensor of any type.
    grad: Variable holding the gradient of type and location matching
        the ``.data``.  This attribute is lazily allocated and can't
        be reassigned.
    requires_grad: Boolean indicating whether the Variable has been
        created by a subgraph containing any Variable, that requires it.
        See :ref:`excluding-subgraphs` for more details.
        Can be changed only on leaf Variables.
    volatile: Boolean indicating that the Variable should be used in
        inference mode, i.e. don't save the history. See
        :ref:`excluding-subgraphs` for more details.
        Can be changed only on leaf Variables.
    creator: Function of which the variable was an output. For leaf
        (user created) variables it's ``None``. Read-only attribute.
    '''
    x = autograd.Variable(torch.ones(2,2), requires_grad = True)
    print(x)
    y = x + 2
    print(y)
    # y was created as the result of the operation and so it has a creator
    print(y.creator)
    z = y * y * 3
    out = z.mean()
    
    print(z, out)
    
    ## Gradients
    
    out.backward()
    print(x.grad)
    # z = 3y^2 = 3(x + 2I)^2 
    # grad of dout/dx |eval x = 1 = 4.5
    
    # I.e2
    x = torch.randn(3)
    x = autograd.Variable(x, requires_grad = True)
    
    y = x * 2
    print(y)
    print(y.data.norm())
    while (y.data.norm() < 1000):
        y = y * 2
    print(y)
    
    gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
    y.backward(gradients)
    
    print(x.grad)
#==============================================================================
# Creating a  Neural Net
#==============================================================================
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel np.Conv2d(input_channels, output_channels, square_convloution)
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # An afine operation y = wx + b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        # Max pooling over a (2,2) window
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2,2))
        # If the size is a square you can only specify a single number
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        # .view is similar to reshape
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size         = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    neuralNet = Net()
    print(neuralNet)
    params = list(neuralNet.parameters())
    print(len(params))
    print(params[0].size()) #conv1's weight
    
    # The input to the forward is an autograd.Variable, therefore the 
    # output will also be a Variable object:
    input = autograd.Variable(torch.randn(1,1,32,32))
#    print(input)
    out = neuralNet(input)
    print(out)
    
    # Zero the gradient buffers of all parameters and backprops with random
    # gradients
    
    neuralNet.zero_grad()
    out.backward(torch.randn(1,10))
    
    
    
#==============================================================================
#     Loss function
#==============================================================================
        
    output      = neuralNet(input)
    target      = autograd.Variable(torch.arange(1,11)) #A dummy target 
    criterion   = nn.MSELoss()
    
    loss        = criterion(output, target)
    print("Loss {0}".format(loss))
    
    # following loss in the backward direction, using the creator attribute, 
    # enables us to see a graph of computations. See below for a few steps.
    print(loss.creator) # MSE loss
    print(loss.creator.previous_functions[0][0]) #linear
    print(loss.creator.previous_functions[0][0].previous_functions[0][0]) #Relu
        
    print('conv1.bias.grad before backward')
    print(neuralNet.conv1.bias.grad)
    
    
    loss.backward()
    
    print('conv1.bias.grad after backward')
    print(neuralNet.conv1.bias.grad)
    
#==============================================================================
#     Updating the weights
#==============================================================================
    # SGD is one of the simplist update rules
    learning_rate = 0.01
    for f in neuralNet.parameters():
        f.data.sub_(f.grad.data*learning_rate)
    # To use different updates, pytorch has the torch.optim package
    # creating an optimizer
      
    optimizer = optim.SGD(neuralNet.parameters(), lr = 0.01)
     
    # in your training loop
    optimizer.zero_grad() # zeroing the gradings
    output = neuralNet(input)
    loss   = criterion(output,target)
    loss.backward()
    optimizer.step()  #does the update. 
      
        
        
        
        
        