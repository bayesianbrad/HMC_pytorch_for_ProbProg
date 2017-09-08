#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:25
Date created:  06/09/2017

License: MIT
'''
import torch
import numpy as np
import Distributions.distributions as dis
from torch.autograd import Variable
from core import VariableCast
class program():
    '''
    Base class for all programs. Values comes in as  tensor and outputs as list
    Unpacking of that list to a tensor is done in the integrator module.
    '''
    def __init__(self):

        self.params  = {'x':None}
    #
    def calc_grad(self, logjoint, values):
        ''' Stores the gradients, grad, in a tensor, where each row corresponds to each the
            data from the Variable of the gradients '''
        assert(isinstance(values, Variable))
        print()
        print(values)
        grad = torch.autograd.grad(logjoint, values, grad_outputs=torch.ones(values.data.size()))[0]
        # For some reason all the gradients are d times bigger than they should be, where d is the dimension

        return grad
    #
    # def tensor_to_list(self,values):
    #     params = []
    #     for value in values:
    #         if isinstance(value, Variable):
    #             temp = Variable(value.data, requires_grad=True)
    #             params.append(temp)
    #         else:
    #             temp = VariableCast(value)
    #             temp = Variable(value.data, requires_grad=True)
    #             params.append(value)
    #     return params
class conjgauss(program):
    def __init__(self):
        super().__init__()

    def generate(self):
        ''' Generates the initial state and returns the samples and logjoint evaluated at initial samples  '''

        ################## Start FOPPL input ##########
        logp = []
        dim = 1
        params = Variable(torch.FloatTensor(1,dim).zero_())
        a = VariableCast(0.0)
        b = VariableCast(2.236)
        normal_object = dis.Normal(a, b)
        x = Variable(normal_object.sample().data, requires_grad = True)
        params = x

        std  = VariableCast(1.4142)
        obs2 = VariableCast(7.0)
        p_y_g_x    = dis.Normal(params[0,:], std)

        logp.append(normal_object.logpdf(params))
        logp.append(p_y_g_x.logpdf(obs2))

        ################# End FOPPL output ############
        dim_values = params.size()[0]

        # sum up all logs
        logp_x_y   = VariableCast(torch.zeros(1,1))
        for logprob in logp:
            logp_x_y = logp_x_y + logprob
        return logp_x_y, params, VariableCast(self.calc_grad(logp_x_y,params)), dim_values
    def eval(self, values, grad= False, grad_loop= False):
        ''' Takes a map of variable names, to variable values . This will be continually called
            within the leapfrog step

        values      -       Type: python dict object
                            Size: len(self.params)
                            Description: dictionary of 'parameters of interest'
        grad        -       Type: bool
                            Size: -
                            Description: Flag to denote whether the gradients are needed or not
        '''
        logp = []  # empty list to store logps of each variable # In addition to foopl input
        assert (isinstance(values, Variable))
        ################## Start FOPPL input ##########
        values = Variable(values.data, requires_grad = True)
        a = VariableCast(0.0)
        b = VariableCast(2.236)
        normal_object = dis.Normal(a, b)

        std  = VariableCast(1.4142)
        obs2 = VariableCast(7.0)
        # Need a better way of dealing with values. As ideally we have a dictionary (hash map)
        # then we say if values['x']
        # values[0,:] = Variable(values[0,:].data, requires_grad  = True)
        p_y_g_x    = dis.Normal(values[0,:], std)

        logp.append(normal_object.logpdf(values[0,:]))
        logp.append(p_y_g_x.logpdf(obs2))

        ################# End FOPPL output ############
        logjoint = VariableCast(torch.zeros(1, 1))

        for logprob in logp:
            logjoint = logjoint + logprob
        # grad2 is a hack so that we can call this at the start
        if grad:
            gradients = self.calc_grad(logjoint, values)
            return gradients
        elif grad_loop:
            gradients = self.calc_grad(logjoint, values)
            return logjoint, gradients
        else:
            return logjoint, values
class linearreg(program):
    def __init__(self):
        super().__init__()

    def generate(self):
        ''' Returns log_joint a tensor.float of size (1,1)
                     params    a Variable FloatTensor of size (#parameters of interest,dim)
                                contains all variables
                     gradients a Variable tensor of gradients wrt to parameters

        '''
        # I will need Yuan to spit out the number of dimensions of  the parameters
        # of interest
        dim   = 1
        logp   = []
        params  = []
        values  = Variable(torch.FloatTensor(1,dim).zero_())
        c23582 = VariableCast(0.0).unsqueeze(-1)
        c23583 = VariableCast(10.0).unsqueeze(-1)
        normal_obj1 = dis.Normal(c23582, c23583)
        x23474 = Variable(normal_obj1.sample().data, requires_grad = True)  # sample
        # append first entry of params
        params.append(x23474)
        p23585 = normal_obj1.logpdf(x23474)  # prior
        logp.append(p23585)
        c23586 = VariableCast(0.0).unsqueeze(-1)
        c23587 = VariableCast(10.0).unsqueeze(-1)
        normal_obj2 = dis.Normal(c23586, c23587)
        x23471 = Variable(normal_obj2.sample().data, requires_grad = True)  # sample
        # append second entry to params
        params.append(x23471)
        p23589 = normal_obj2.logpdf(x23471)  # prior
        logp.append(p23589)
        c23590 = VariableCast(1.0).unsqueeze(-1)
        x23591 = x23471 * c23590 + x23474 # some problem on Variable, Variable.data

        # x23592 = Variable(x23591.data + x23474.data, requires_grad = True)

        c23593 = VariableCast(1.0).unsqueeze(-1)
        normal_obj2 = dis.Normal(x23591, c23593)

        c23595 = VariableCast(2.1).unsqueeze(-1)
        y23481 = c23595
        p23596 = normal_obj2.logpdf(y23481)  # obs, log likelihood
        logp.append(p23596)
        c23597 = VariableCast(2.0).unsqueeze(-1)

        # This is highly likely to be the next variable
        x23598 = x23471.mm(c23597) + x23474
        # x23599 = torch.add(x23598, x23474)
        c23600 = VariableCast(1.0).unsqueeze(-1)
        # x23601 = dis.Normal(x23599, c23600)

        normal_obj3 = dis.Normal(x23598, c23600)
        c23602 = VariableCast(3.9).unsqueeze(-1)
        y23502 = c23602
        p23603 = normal_obj3.logpdf(y23502)  # obs, log likelihood
        logp.append(p23603)
        c23604 = VariableCast(3.0).unsqueeze(-1)
        x23605 = x23471.mm(c23604)
        x23606 = torch.add(x23605, x23474)
        c23607 = VariableCast(1.0).unsqueeze(-1)
        normal_obj4 = dis.Normal(x23606, c23607)
        c23609 = VariableCast(5.3).unsqueeze(-1)
        y23527 = c23609
        p23610 = normal_obj4.logpdf(y23527)  # obs, log likelihood
        logp.append(p23610)
        p23611 = Variable(torch.zeros(1,1))
        for logprob in logp:
            p23611 = logprob + p23611
        for i in range(len(params)):
            if i == 0:
                values = params[i]
            else:
                values = torch.cat((values, params[i]), dim=0)
        dim_values = values.size()[0]
        # dim_values = values.size()[0]
        # return E from the model
        # Do I want the gradients of x23471 and x23474? and nothing else.
        grad = torch.autograd.grad(p23611, params, grad_outputs=torch.ones(values.size()))
        # For some reason all the gradients are d times bigger than they should be, where d is the dimension
        gradients = Variable(torch.Tensor(values.size()))
        for i in range(len(params)):
            gradients[i, :] = 1 / len(params) * grad[i][0].data.unsqueeze(0)  # ensures that each row of the grads represents a params grad
        return p23611,values, gradients, dim_values

    def eval(self, values, grad=False, grad_loop=False):
        logp   = []
        assert(isinstance(values, Variable))
        values = Variable(values.data)
        for i in range(values.data.size()[0]):
            values[i,:]  = Variable(values[i,:].data, requires_grad = True)
        c23582 = VariableCast(0.0).unsqueeze(-1)
        c23583 = VariableCast(10.0).unsqueeze(-1)
        normal_obj1 = dis.Normal(c23582, c23583)

        x23474 = values[0,:].unsqueeze(-1)# sample
        # append first entry of params
        p23585 = normal_obj1.logpdf(x23474)  # prior
        logp.append(p23585)
        c23586 = VariableCast(0.0).unsqueeze(-1)
        c23587 = VariableCast(10.0).unsqueeze(-1)
        normal_obj2 = dis.Normal(c23586, c23587)
        x23471 = values[1,:].unsqueeze(-1)# sample
        p23589 = normal_obj2.logpdf(x23471)  # prior
        logp.append(p23589)
        c23590 = VariableCast(1.0).unsqueeze(-1)
        x23591 = x23471 * c23590 + x23474  # some problem on Variable, Variable.data

        # x23592 = Variable(x23591.data + x23474.data, requires_grad = True)

        c23593 = VariableCast(1.0).unsqueeze(-1)
        normal_obj2 = dis.Normal(x23591, c23593)

        c23595 = VariableCast(2.1).unsqueeze(-1)
        y23481 = c23595
        p23596 = normal_obj2.logpdf(y23481)  # obs, log likelihood
        logp.append(p23596)
        c23597 = VariableCast(2.0).unsqueeze(-1)

        # This is highly likely to be the next variable
        x23598 = torch.mul(x23471, c23597) + x23474
        # x23599 = torch.add(x23598, x23474)
        c23600 = VariableCast(1.0).unsqueeze(-1)
        # x23601 = dis.Normal(x23599, c23600)

        normal_obj3 = dis.Normal(x23598, c23600)
        c23602 = VariableCast(3.9).unsqueeze(-1)
        y23502 = c23602
        p23603 = normal_obj3.logpdf(y23502)  # obs, log likelihood
        logp.append(p23603)
        c23604 = VariableCast(3.0).unsqueeze(-1)
        x23605 = torch.mul(x23471, c23604)
        x23606 = torch.add(x23605, x23474)
        c23607 = VariableCast(1.0).unsqueeze(-1)
        normal_obj4 = dis.Normal(x23606, c23607)
        c23609 = VariableCast(5.3).unsqueeze(-1)
        y23527 = c23609
        p23610 = normal_obj4.logpdf(y23527)  # obs, log likelihood
        logp.append(p23610)
        p23611 = Variable(torch.zeros(1, 1))
        for logprob in logp:
            p23611 = logprob + p23611
        if grad:
            gradients = 1 / values.size()[0]  * torch.autograd.grad(p23611, values, grad_outputs=torch.ones(values.size()))[0].data
            # For some reason all the gradients are d times bigger than they should be, where d is the dimension
            print(gradients)
            return Variable(gradients)
        elif grad_loop:
            gradients = 1 / values.size()[0] * \
                        torch.autograd.grad(p23611, values, grad_outputs=torch.ones(values.size()))[0].data
            # For some reason all the gradients are d times bigger than they should be, where d is the dimension
            return p23611, Variable(gradients)
        else:
            return p23611, values

class conditionalif(program):
    def __init__(self):
        '''Generating code, returns  a map of variable names / symbols '''
        self.params = {'x': None}

    def generate(self):
        logp = []  # empty list to store logps of each variable
        a = VariableCast(0.0)
        b = VariableCast(1)
        c1 = VariableCast(-1)
        normal_obj1 = dis.Normal(a, b)
        x = Variable(normal_obj1.sample().data, requires_grad=True)
        logp_x = normal_obj1.logpdf(x)

        if torch.gt(x.data, torch.zeros(x.size()))[0][0]:
            y = VariableCast(1)
            normal_obj2 = dis.Normal(b, b)
            logp_y_x = normal_obj2.logpdf(y)
        else:
            y = VariableCast(1)
            normal_obj3 = dis.Normal(c1, b)
            logp_y_x = normal_obj3.logpdf(y)

        logp_x_y = logp_x + logp_y_x

        return logp_x_y, x, VariableCast(self.calc_grad(logp_x_y, x))

        # sum up all logs
        logp_x_y = VariableCast(torch.zeros(1, 1))
        for logprob in logp:
            logp_x_y = logp_x_y + logprob
        return logp_x_y, x, VariableCast(self.calc_grad(logp_x_y, x))

    def eval(self, values, grad=False, grad_loop=False):
        ''' Takes a map of variable names, to variable values '''
        params = self.tensor_to_list(values)
        a = VariableCast(0.0)
        b = VariableCast(1)
        c1 = VariableCast(-1)
        normal_obj1 = dis.Normal(a, b)
        values = Variable(values.data, requires_grad=True)
        logp_x = normal_obj1.logpdf(values)
        # else:
        #     x = normal_object.sample()
        #     x = Variable(x.data, requires_grad = True)
        if torch.gt(values.data, torch.zeros(values.size()))[0][0]:
            y = VariableCast(1)
            normal_obj2 = dis.Normal(b, b)
            logp_y_x = normal_obj2.logpdf(y)
        else:
            y = VariableCast(1)
            normal_obj3 = dis.Normal(c1, b)
            logp_y_x = normal_obj3.logpdf(y)

        logjoint = Variable.add(logp_x, logp_y_x)
        if grad:
            gradients = self.calc_grad(logjoint, values)
            return VariableCast(gradients)
        elif grad_loop:
            gradients = self.calc_grad(logjoint, values)
            return logjoint, VariableCast(gradients)
        else:
            return logjoint, values