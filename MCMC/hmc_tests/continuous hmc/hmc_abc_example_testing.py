#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt

def main():
    global count
    count       = 0
    num_samples = 10000
    nvars       = 3 
    ndim        = 3
    
    sample1    = torch.Tensor(num_samples, ndim)
    variables  = torch.Tensor(1)
    variables  = 
    

    print("Starting to draw samples")
    # TODO should requires_grad=True below?
    initial_x = Variable(torch.Tensor([[0,20,0]]), requires_grad=True)
    sample1[0] = initial_x.data
    for i in range(num_samples-1):
        sample,count = hmc(initial_x,
                log_posterior = abc_log_posterior,
                step_size = torch.min(torch.Tensor(1).uniform_(0.01,0.02)),
                num_steps = torch.Tensor(1).uniform_(15,24).int()[0],
                ndim      = ndim,
                count     = count)
        sample1[i+1] = sample.data

    print("success")

    return sample1

def log_1dgaussian_pdf(x, mu, std):
    if not isinstance(mu, Variable):
        mu = Variable(mu, requires_grad = False)
    if not isinstance(std, Variable):
        std = Variable(std, requires_grad = False)

    # we don't care about constant factors so it reduces to
    return -(x-mu)*(x-mu)/(std*std)

def abc_log_posterior(x, ndim=3, grad = False):
    """
    x : torch.autograd.Tensor or torch.autograd.Variable of size 3 - (a, b, c)
    """
    if not isinstance(x, Variable):
        x = Variable(x, requires_grad = True)

    # x = np.arange(3, dtype=float).reshape((3,1))
    # x = torch.Tensor(x)
    # x = Variable(x, requires_grad = True)

    a, b, c = torch.chunk(x, 3, dim=1)

    zero = torch.Tensor([0])
    one = torch.Tensor([1])
    twenty = torch.Tensor([20])

    dlogp = \
        log_1dgaussian_pdf(x=a, mu=zero, std=one) * \
        log_1dgaussian_pdf(x=b, mu=twenty, std=one) * \
        log_1dgaussian_pdf(x=c, mu=a, std=b)

    if grad:
        dlogp.backward()
        dlogp_dx = x.grad.data
        return dlogp_dx
    else:
        return dlogp.data


def kinetic_energy(momentum, grad = False):
    """Kinetic energy of the current momentum (assuming a standard Gaussian)
        (x dot x) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}
    Parameters
    ----------
    momentum : torch.autogram.Variable
        Vector of current momentum
    Returns
    -------
    kinetic_energy : float
    """
    # mass matrix
    P = Variable(momentum, requires_grad = True)
    M = Variable(torch.eye(momentum.size()[1]), requires_grad = False)
    K = 0.5 * P.mm(M).mm(torch.transpose(P,0,1))
    if grad:
        K.backward()
        return P.grad.data
    else:
        return K.data
#    return 0.5 * torch.dot(momentum, momentum)

def hamiltonian(position, momentum, energy_function,ndim):
    """Computes the Hamiltonian  given the current postion and momentum
    H = U(x) + K(p)
    U is the potential energy and is = -log_posterior(x)
    Parameters
    ----------
    position        :torch.autograd.Variable, we requires its gradient.
                     Position or state vector x (sample from the target
                     distribution)
    momentum        :torch.Tensor \mathbb{R}^{1 x D}. Auxiliary momentum
                     variable
    energy_function :Function from state to position to 'energy'= -log_posterior

    Returns
    -------
    hamitonian : float
    """
    U = energy_function(position,ndim)
    T = kinetic_energy(momentum)
    return U + T

def leapfrog_step(x0, p0,log_posterior, step_size, num_steps,  ndim):
    '''Performs the leapfrog steps of the HMC for the specified trajectory
    length, given by num_steps'''

    # Start by updating the momentum a half-step
    p = p0 + 0.5 * step_size * log_posterior(x0, ndim, grad = True)
    # Initalize x to be the first step
    x0.data = x0.data + step_size * kinetic_energy(p, grad = True)
    # If the gradients are not zeroed then they will blow up. This leads
    # to an exponentiatin kinetic and potential energy. As the positon
    # and momentum increase.
    x0.grad.data.zero_()
    for i in range(num_steps-1):
        # Compute gradient of the log-posterior with respect to x
        # Update momentum
        p = p + step_size * log_posterior(x0, ndim, grad = True)

        # Update x
        x0.data = x0.data + step_size *  kinetic_energy(p, grad = True)
        x0.grad.data.zero_()

    # Do a final update of the momentum for a half step

    p = p + 0.5 * step_size * log_posterior(x0,ndim,grad = True)

    # return new proposal state
    return x0, p

def hmc(initial_x, step_size, num_steps,log_posterior, ndim, count):
    """Summary
    Parameters
    ----------
    initial_x : torch.autograd.Variable
        Initial sample x ~ p
    step_size : float Step-size in Hamiltonian simulation
    num_steps : int  Number of steps to take in leap frog
    log_posterior : str
        Log posterior (unnormalized) for the target distribution
        takes ndim, grad(bool)
    Returns
    -------
    sample :
        Sample ~ target distribution
    """
    p0 = torch.randn(initial_x.size())
    x, p = leapfrog_step(initial_x,
                      p0,
                      step_size=step_size,
                      num_steps=num_steps,
                      log_posterior=log_posterior,
                      ndim = ndim)

    orig = hamiltonian(initial_x, p0, log_posterior,ndim)
    current = hamiltonian(x, p, log_posterior,ndim)
    alpha = torch.min(torch.exp(orig - current))
    p_accept = min(1,alpha)
    if p_accept > np.random.uniform():
        count = count + 1
        return x, count
    else:
        return initial_x,count

samples = main()
print("Means:")
print(np.mean(samples.numpy(), axis=0))
print("Standard deviations:")
print(np.std(samples.numpy(), axis=0))
