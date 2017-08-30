#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:12:18 2017

@author: bradley

Borg class pattern

Will be incredibly useful for HMC !!!!!!!!!
"""

class Borg:
    ''' Borg class making class attributes global'''
    _shared_state = {} # Attribute dictionary
    
    def __init__(self):
        # Make it an attribute dictionary
        self.__dict__  = self._shared_state
        
class Singleton(Borg): # Inherits form the Borg class
    '''This class now shares all its attributes amount its various instances'''
    # Essentially makes singleton object 'a global var'
    def __init__(self, **kwargs):
        Borg.__init__(self)
        # update the attribute dictionaty by inserting a new key-value pair
        self._shared_state.update(kwargs)
    def __str__(self):
        #Returns the attribute disctionary for printing
        return str(self._shared_state)


def sample_function(number, count):
    for i in range(count):
        number = number + 1
    return number

x = Singleton(HTTP = 'hyper text transfer protocal', samples = 1000, burn_in = 100, function = sample_function)
print(x)
y = x.function(0,3)
print(y)
#y  =Singleton(SNMP = 'Simple Network Management Protocol')
#print(y)