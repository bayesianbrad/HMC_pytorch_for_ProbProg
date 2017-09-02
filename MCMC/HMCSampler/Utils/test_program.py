#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:51
Date created:  01/09/2017

License: MIT
'''

import torch
from torch.autograd import Variable
from core import VariableCast
from program import program
from test_KE import KEnergy
# TO DO: Check how to call a class method within  a class
def calc_grad(logjoint, values):
    ''' Stores the gradients, grad, in a tensor, where each row corresponds to each the
        data from the Variable of the gradients '''
    # Assuming values is a dictionary we could extract the values into a list as follows
    # if isinstance(dict, values):
    #     self.params = list(values.values())
    # else:
    #     self.params = values
    grad = torch.autograd.grad([logjoint], [values], grad_outputs=torch.ones(values.data.size()))
    # note: Having grad_outputs set to the dimensions of the first element in the list, implies that we believe all
    # other values are the same size.
    gradients = torch.Tensor(len(values), values.data.size())
    for i in range(len(values)):
        gradients[i][:] = grad[i][0].data.unsqueeze(
            0)  # ensures that each row of the grads represents a params grad
    return gradients
def test():
    prog_obj = program()
    logjointOrig, values_init, init_gradient  = prog_obj.generate()
    print(logjointOrig, values_init)
    print(init_gradient)
    ham_orig                   = fake_ham(logjointOrig)
    #
    # # in the future we would have to change this line so that
    # # if values is a dictionary then, we can generate a
    # # momentum with the right
    p0         = VariableCast(torch.randn(values_init.size()))
    kinetic_obj = KEnergy(p0)
    values     = values_init
    print('******** Before ********')
    print(p0)
    # first half step
    print(type(p0))
    print(type(init_gradient))
    p = p0 + 0.5 *  init_gradient
    print('******* Before ******')
    print(values)
    print(p)
    print()
    for i in range(10-1):
        print('Iter :', i )
        p      = p + 0.5 * prog_obj.eval(values,grad=True)
        values = values + 0.5 *  kinetic_obj.gauss_ke(p, grad = True)
        print('**** Inter ****')
        print(p.data)
        print(values.data)
    print('******** After ********')
    print(values)
    print(p)


def fake_ham(logjoint):
    return torch.exp(logjoint + 2.0)

test()