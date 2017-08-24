##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Thu Aug 17 11:28:41 2017
#
#@author: bradley
#"""
#
import torch
import numpy as np
from torch.autograd import Variable
import scipy.stats as ss
import math
#
#def kinetic_fn(p, mom  ='Gauss', grad = False):
#    """Kinetic energy of the current momentum (assuming a standard Gaussian)
#        (x dot x) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}
#    Parameters
#    ----------
#    p    : torch.autogram.Variable
#          Vector of current momentum
#    mom  : String representing type of momentum to use
#    grad : bool
#    Returns
#    -------
#    kinetic_energy : float
#    """
#    # mass matrix
#    P = Variable(p, requires_grad = True)
#    M = Variable(torch.eye(p.size()[1]), requires_grad = False)
#    K = 0.5 * P.mm(M).mm(torch.transpose(P,0,1))
#    if grad:
#        K.backward()
#        return P.grad.data
#    else:
#        return K.data
#def log_potential_fn(observe, mean,cov, grad = False):
#    """Evaluate the unormalized log posterior from a zero-mean
#    Gaussian distribution, with the specifed covariance matrix
#    
#    Parameters
#    ----------
#    x :  In general should be torch.autograd.Variable. Else, will instantiate
#         one. 
#    Sample ~ target distribution
#    cov_inverse : torch.autograg.Variable N x N 
#
#
#    Returns
#    -------
#    logp : float, log(exp(-x.T * A * x)) "Log of Normal"
#        Unormalized log p(x)
#    dlogp_dx : float: Gradient of the log. 
#    """
#    if not isinstance(x, Variable):
#        X = Variable(x, requires_grad = True)
#    else:
#        X = x
#    det = np.abs(np.linalg.det(cov.numpy()))
#    constant = np.sqrt((2*np.pi*np.log(det))**x.size()[1]/2)
#    print(constant)
#    xAx = -0.5*X.mm(cov_inverse).mm(torch.transpose(X,0,1)) - constant
#    if grad:
#        xAx.backward()
#        dlogp_dx = X.grad.data
#        return dlogp_dx 
#    else:
#        return xAx.data
torch.manual_seed(123445)

def opt3():
    X   = Variable(torch.Tensor([7]), requires_grad = False)
    x1  = Variable(torch.randn(1,1), requires_grad = True)
    cov_inv = Variable(torch.eye(1), requires_grad = False)
    log_normal_pdf = -0.5*(X-x1)*cov_inv*(X-x1)
    log_normal_pdf.backward()
    print(x1.grad.data)

def opt4():
    # Variables
    x1  = Variable(torch.normal(mean = 0.0, std  = torch.Tensor([10])), requires_grad = True)
    x2  = Variable(torch.normal(mean = 0.0, std = torch.Tensor([10])), requires_grad = True)
    # Std
    std1 = Variable(torch.Tensor([1.0]), requires_grad = False)
    std2 = Variable(torch.Tensor([1.0]), requires_grad = False)
    # Observes
    observe1  = Variable(torch.Tensor([2.1]), requires_grad = False)
    observe2  = Variable(torch.Tensor([3.9]), requires_grad  =False)
    # regression variab;es
    y1 = normal_pdf(observe1, x1 + x2, std1 )
    y2 = normal_pdf(observe2, 2*x1 + x2, std2)
    # value + pdf of all random variables in the program. 
    # Random variable class , it must account for two computation graphs
    # one of the log pdfs of all random variables
    # and the value of the random variable
    # true grad
    true1 = (x1.data + x1.data*x2.data)*y1.data
#    true2 =-0.5*(2*(x1.data + x2.data) - 2*(x1.data + x2.data))*y1.data
    y1.backward()
    dy1_dx1 = x1.grad.data.clone()
    dy1_dx2 = x2.grad.data.clone()
    print('grad dy1_dx1', dy1_dx1)
    print('grad dy1_dx2', dy1_dx2)
    print()
    print('True grad', true1)
    x1.grad.data.zero_()
    x2.grad.data.zero_()
    y2.backward()
    dy2_dx1 = x1.grad.data.clone()
    dy2_dx2 = x2.grad.data.clone()
    print()
    print('grad dy2_dx1', dy2_dx1)
    print('grad dy2_dx2', dy2_dx2)
    print()
#    print('True grads',true1, true2)

def normal_pdf(observed, mean, std):
    pi        = np.pi
    constant  = torch.sqrt(2*torch.abs(std)**2 * pi)
    print('Constant' , constant)
    return constant*torch.exp(-0.5*(observed - mean)*(std**-1)*(observed-mean))
def normal_pdf_data(observed, mean, std):
    pi        = np.pi
    constant  = torch.sqrt(2*torch.abs(std.data)**2 * pi)
    return constant*torch.exp(-0.5*(observed.data - mean.data)*(1/std.data)*(observed.data-mean.data))
opt4()