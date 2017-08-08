#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:47:46 2017

@author: bradley
"""

class hmc(object):
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)