#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:11:53 2017

@author: bradley
"""
import torch 
import numpy as np
from torch.autograd import Variable

# Data - Our discrete variables of the Jolly Seber model 
n  = torch.from_numpy(np.array([54, 146, 169, 209, 220, 209, 250, 176, 172, 127, 123, 120, 142]))
R  = torch.from_numpy(np.array([54, 143, 164, 202, 214, 207, 243, 175, 169, 126, 120, 120]))
r  = torch.from_numpy(np.array([24, 80, 70, 71, 109, 101, 108, 99, 70, 58, 44, 35]))
m0 = torch.from_numpy(np.array([0, 10, 37, 56, 53, 77, 112, 86, 110, 84, 77, 72, 95])) # Exclude m_1 = 0
m  = torch.from_numpy(np.array([10, 37, 56, 53, 77, 112, 86, 110, 84, 77, 72, 95]))
z  = torch.from_numpy(np.array([14, 57, 71, 89, 121, 110, 132, 121, 107, 88, 60, 0])) # Exclude z_1 = 0
u  = n - m0
T  = len(n)