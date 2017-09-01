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
def test():
    prog_obj = program()
    logjointOrig, values_init  = prog_obj.generate()
    initial_grad               = program.calc_grad(logjoint= logjointOrig, values = values_init)
    kinetic_obj                = KEnergy()
    ham_orig                   = fake_ham(logjointOrig)

    # in the future we would have to change this line so that
    # if values is a dictionary then, we can generate a
    # momentum with the right
    p0         = VariableCast(torch.randn(values_init.size))
    values     = values_init
    print('******** Before ********')
    print(p0)
    p = p0 + 0.5 *  initial_grad
    print(values)
    print()
    for i in range(10-1):
        p      = p + 0.5 * prog_obj.eval(values, grad = True)
        values = values + 0.5 *  kinetic_obj.gauss_ke(p, grad = True)
    print('******** After ********')
    print(values)
    print(p)


def fake_ham(logjoint):
    return torch.exp(logjoint + 2.0)

test()