#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:05:14 2017

@author: bradley

C+DHMC in pytorch

Based on the work of Nishimura et al.

# Notes:
     n_params and n_disc are inputted in to the sampler in a swapped
     order.
     We do not pass f_update in to DHMC sampler, hoping to do that via 
     pytorch's autograd
     Add kinetic function that will take M and p. Creates Variables in the 
     fucntion and inverts M. May invert M on intiation of object. 
     Removed grad input from gauss_laplace_leapfrog
     Ensure all x,p and gradients are \mathbb{R}^{D x 1}  as that allows for
     efficient slicing or the tensor. As we would do in Numpy. 
     

     

"""
import torch
import numpy as np
import math
from torch.autograd import Variable
torch.utils.backcompat.broadcast_warning.enabled=True

class Potential(object):
    ''' Is told what type of potential we are calulating, discrete or 
    continous and ouputs the potential and required gradients if needed. 
    Parameters
    ----------
    x             - A list containing our parameters of interest
    
    jnt_density   - Will be a  function object that contains the entire 
                    joint and allows us to evaluate it 
    grad          - Bool, will be true or false depending on whether 
                    the gradient is needed
    continuous    - Bool, default True. States whether we require the 
                    continous potential
    discrete      - Bool, default False. States whether we require the 
                    discrete potential.
    Output
    --------
    
    potential object. 
    '''
    def __init__(self, x, jnt_density, grad, continuous = True, discrete = False ):
        self.x              = x
        self.jnt_density    = jnt_density
        self.grad           = grad
        self.continuous     = continuous
        self.discrete       = discrete
    
    def _determine_parameters(self, x):
        ''' Extracts the parameters of interest from x, so that 
        we can operate on them accordingly. Assume params are stored
        row wise.
        
        Parameters
        ----------
        x  - N x D tensor. N - Number of params, D dimension of data
        
        Output
        --------
        params - A 1 x N list of params of type variable
        
        '''
        no_params = x.size()[0]
                
        
    def disc_potential(self):
        '''To do'''
    def cont_potential(self, **kwargs):
        '''To do'''
    def log_potential_fn(self, x, jnt_density ):
        '''Evaluate the unormalized log posterior, the joint distribution.
        
        Parameters
        ----------
        x : contains a
    
       Output
        -------
        logp : float, unormalized log p(x)
        dlogp_dx : float: Gradient of the log. 
        
        '''
        if not isinstance(x, Variable):
            X = Variable(x, requires_grad = True)
        else:
            X = x
        xAx = -0.5*X.mm(cov_inverse).mm(torch.transpose(X,0,1))
        if grad:
            xAx.backward()
            dlogp_dx = X.grad.data
            return dlogp_dx 
        else:
            return xAx.data
        
class DHMCSampler(object):
    '''Discountinous HMC sampler. Takes as input an object, and outputs
    the samples. 
    
    Parameters:
    -----------
    potenital    -   A function(x, req_grad). Type: function object. 
                     Containts the log of the potential. Which is the log of 
                     our target distribution of interest. The gradient of the
                     discrete parameters is zero.**Has the ability to 
                     return gradients too, if we are evaluating continous
                     parameters.
    potential_disc - A function(theta, dtheta, index, aux).
                     Computes the difference in log of the target 
                     distribution when x[index] is modifed by dx.
                     aux represents whatever quantity is saved from the 
                     the previous call to potential or potential_disc. See 
                     gauss_laplace_leapfrog class method. 
    kinetic        - A function(p, M, mom, req_grad). The function of 
                     the kinetic energy, returns gradient if required for
                     the required momentu i.e Gauss or Laplace. 
    n_params       - A scalar N \in \mathbb{N}. The total  number of continous 
                     and discontinous parameters. 
    n_disc         - A scalar Q. The total number of discontinous parameters
    scale          - A torch.Tensor \mathcal{R}^{1 x N}. Required heuristic for
                  constructing the mass matrix of the Laplace momentum terms
   Outputs:
   -----------
   Sampler class
    '''
    def __init_(self, potential,potential_disc, kinetic, n_params,\
                n_disc, scale = None):
        if scale is None:
            scale = torch.ones((n_params,1))
        # Scale is used to set the mass matrix for a Laplace momentum. 
        self.M              = torch.div(torch.ones(1),\
                                        torch.cat(scale[:-n_disc]**2, scale[-n_disc:]))
        self.n_params       = n_params
        self.n_disc         = n_disc
        self.potential      = potential
        self.potential_disc = potential_disc
        self.kinetic        = kinetic
    def gauss_laplace_momentum(self, potential,potential_disc, kinetic,\
                               stpsze, x0, p0, logp, grad, aux, n_disc = 0,\
                               M  = None):
        '''Performs one numerical integration step of the DHMC  mixed 
        integrator, Alogirthm 2 from Nishimura et al. 2017. 
        
        Parameters
        ----------
        
        stpsze -  A scalar representing step size.  
        x0     -  A torch.Tensor \in mathbb{R}^{D x 1}. Represents the intial 
                  starting value. 
        p0     -  A torch.Tensor \in mathbb{R}^{D x 1}. Represents the intial,
                  sampled momentum. 
        logp   -  A function representing the log of the posterior. 
        grad   -  A torch.Tensor, representing the gradient of the potential 
                  energy evaluated at the specified point. 
                  ** May not need this if we include the functionality in the
                  potential function. **
        M      -  A torch.Tensor M \in mathb{R}^{D x D}. Represents the 
                  diagonal of the mass matrix. 
        
        Outputs
        --------
        
        x        - A torch.Tensor \mathbb{R}^{D x 1}. The proposed x
        p        - A torch.Tensor \mathbb{R}^{D x 1}. The proposed p
        logp     - A torch.Tensor \mathbb{R}^{1 x 1}. As previously defined
        gradcont - A troch.Tensor \mathbb{R}^{D x 1}. A tensor of the
                   gradients of the continous parameters
        aux      - Output from the potential. ** may not be needed. 
        '''
        if M is None:
            M  = torch.ones(x0.size()[0])
        
        x  = x0.clone()
        p  = p0.clone()
        # performs first half step on continous parameters
        p[:-n_disc] = p[:-n_disc] + 0.5*stpsze*potential(p[:-n_disc],grad = True)
        if n_disc == 0:
            x = x + stpsze*kinetic(p,M, mom = 'Gauss', grad = True)
        else:
            # Update continous parameters if any
            if self.n_param != self.n_disc:
                x[:-n_disc] = x[:-n_disc] + stpsze * 0.5 * kinetic(p[:-n_disc],M[:-n_disc], mom = 'Gauss', grad = True)
                logp,aux  = potential(x, grad = False)
                gradcont  = potential(x, grad = True)
            # Update discrete parameters
            if math.isinf(logp[0]):
                 return x, p, grad, logp, n_feval, n_fupdate #  not sure what to do here line 149
            # creates a function to permute 'J' indices. I.e J is the
            # the discontinous set of parameters. Line 3 of Algorithm 2
            coord_order = x.size()[0] - n_disc + np.random.permutation(n_disc)
            for index in coord_order:
                # calls on Algorithm 1 from Nishimura et al.
                x, p, logp, aux = self.coordwise_int(self, potential_disc,\
                                                     aux, index, x, p, M,\
                                                     stpsze, logp)
            x[:-n_disc] = x[:-n_disc] + stpsze * 0.5 * kinetic(p[:-n_disc],M[:-n_disc],mom = 'Gauss',grad = True)
            
        if self.n_param != self.n_disc:
            logp     = potential(x, grad = 'False')
            gradcont = potential(x, grad = 'True')
        
        return x, p, gradcont, logp, aux
    
    def coordwise_int(self, potential_disc, aux, index, x, p, M, stpsze, logp):
        '''Performs the coordinatewise integrator update step, algorithm 1. 
        Using a Laplace momentum.
        
        Parameters
        -----------
        
        index   - Scalar - Passed from gauss_laplace_momentum method 
                 in order to do loops. 
                 
        Outputs
        -------
        x       - 
        p       -
        logp    - 
        aux     - 
        '''
        p_sign      = math.copysign(1.0, p[index][0])
        # derivative of Laplace momentum
        dx          = stpsze*torch.div(1.0,M[index][0]) 
        logp_diff,_ = potential_disc(x, dx,index, aux ) 
        # code assumes dU is a 1 x 1 torch.Tensor
        dU          = - logp_diff
        # if the momentum is enough to carry us over the step then do...
        # else.
        if torch.abs(torch.div(p[index],M[index]))[0] > dU[0]:
            p[index][0] =  p[index][0]  - p_sign * M[index] * dU
            x[index][0] =  x[index][0] +  dx
            logp        = logp + logp_diff
            aux         = _ # may not be needed
        else:
            p[index]    = - p[index]
        return x, p, logp, aux
    
    def hmc(self, stpsze, L, x0, logp0, grad0, aux0, potential, kinetic):
        '''
        ****** should not need grad0 or aux0 when code goes to next stage
        
        A standard hmc proposal scheme to use with any kinetic or porential
        energy that statisfies the properties of being both reversible and
        volume-preserving. 
        
        Parameters
        ----------
        L - Scalar. Represents the number of integrator steps to take. 
        
        Outputs
        ---------
        x          - torch.Tensor \mathbb{R}^{D x 1}. Proposed x
        logp       - torch.Tensor \mathbb{R}^{D x 1}. Log of the target 
                     distribution
        *grad      - * can just be returned via the potential function
        *aux
        acceptprob
        *count 
        '''
        
        p        = self.gen_momentum()
        hamorig  = self.hamiltonian(logp0, p, potential, kinetic)
        
        count    = 0
        # performs first full integrator step
        x, p,  grad, logp, aux \
            = self.integrator(stpsze, x0, p, logp0, grad0, aux0)
        count    = count + 1
        for i in range(1,L):
            if math.isinf(logp[0]):
                break
            x, p, grad, logp, aux = self.integrator(stpsze, x, p, logp, grad, aux)
            count = count + 1
        # the hamiltonian 'joint' of the proposed x and p. 
        hamprop  = self.hamiltonian(logp, p, potential, kinetic)
        alpha    = torch.min(torch.exp(hamorig - hamprop))
        p_accept = min(1,alpha)
        if p_accept < np.random.uniform():
            x    = x0
            logp = logp0
            grad = grad0
         
        return x, logp, grad, aux, alpha,  count
        
    def gen_momentum(self):
        '''
        Generates random momentum based on the heuristics provided in 
        Nishimura et al. 2017. i.e pcont = √m_{i}*\matcal{N}(0,1)
        and pdisc = m_{i} * \mathcal{L}(0,1) (lapalce distrubution with mean
        0 and var 1. )
        '''
        p_cont = torch.sqrt(self.M[:-self.n_disc]) \
        * torch.randn(self.n_param - self.n_disc,1)
        p_disc = self.M[-self.n_disc:] * torch.from_numpy(np.random.laplace(size=self.n_disc))
        return torch.cat((p_cont, p_disc),)
    
    def hamiltonian(self, x, p, potential, kinetic):
        '''
        Computes the Hamiltonian  given the current postion and momentum
        H = U(x) + K(p)
        U is the potential energy and is = -log_posterior(x)
        
        Parameters
        ----------
        x    - torch.Tensor.
        p    - torch.Tensor. 
            Auxiliary momentum variable
        energy_function
            Function from state to position to 'energy'
             = -log_posterior
        
        Output
        -------
        hamitonian - float
        '''
        U     = potential(x, grad = False) 
        Tcont = kinetic(p[:-self.n_disc,1], self.M, mom = 'Gauss',  grad = False)
        Tdisc = torch.abs(torch.transpose(p[-self.n_disc:,1],0,1).mm(torch.div(torch.ones(1,1),\
                                          self.M[-self.n_disc:,1])).mm(p[-self.n_disc:,1]))
        return U + Tcont + Tdisc
    # Integrator and kinetic energy functions for the proposal scheme. The
    # class allows any reversible dynamics based samplers by changing the
    # 'integrator', 'random_momentum', and 'compute_hamiltonian' functions.
    def integrator(self, stpsze, x0, p0, logp, grad, aux):
        return self.gauss_laplace_leapfrog(
            self.potential, self.potential_disc, stpsze, x0, p0,\
            logp, grad, aux, self.n_disc, self.M)
    def run_sampler(self, x0, stpsze_range, L_range, n_burnin, n_sample, seed=None, n_update=10):
        """Run DHMC and return samples and some additional info.
        Note: stpsze_range is a torch tensor ,like wise for L_range"""

        np.random.seed(seed)

        # Run HMC.
        x = x0
        n_per_update = math.ceil((n_burnin + n_sample) / n_update)
        pathlen_ave  = 0
        # Due to the way torch generates tensors when slicing we 
        # have to reverse the tuple arguments in sample
        samples      = torch.zeros(x.size()[0],n_sample + n_burnin)
        logp_samples = torch.zeros(n_sample + n_burnin)
        accept_prob  = torch.zeros(n_sample + n_burnin)

#        tic = time.process_time()  # Start clock
        logp, grad, aux = self.potential(x)
        for i in range(n_sample + n_burnin):
            stpsze = np.random.uniform(stpsze_range[0], stpsze_range[1])
            # may have to convert nstep_range from torch to np array or 
            # viceversa. Likewise for L.
            L      = np.random.randint(L_range[0], L_range[1] + 1)
            theta, logp, grad, aux, accept_prob[i], pathlen \
                = self.hmc(stpsze, L, x, logp, grad, aux)
            pathlen_ave = i / (i + 1) * pathlen_ave + 1 / (i + 1) * pathlen
            samples[:, i] = x
            # CHECK THIS TOMORROW THAT LOGP_SAMPLES[I][0] WORKS5493875789348587
            logp_samples[i][0] = logp
            if (i + 1) % n_per_update == 0:
                print('{:d} iterations have been completed.'.format(i + 1))

#        toc = time.process_time()
#        time_elapsed = toc - tic
        print(('The average path length of each DHMC iteration was '
               '{:.2f}.'.format(pathlen_ave)))

        return samples, logp_samples, accept_prob, pathlen_ave