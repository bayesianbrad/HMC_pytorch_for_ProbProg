#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:20:07 2017

@author: bradley
"""
from matplotlib import pyplot as plt
def euler_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
    # Assumes mass is constant for all particles
    p_prev = init_p
    q_prev = init_q
    
    
    for i in range(steps):
        # Plot phase space
        plt.figure(1)
        plt.plot(q_prev,p_prev,'r*')
        # Calculate next p and q values
        p_next = p_prev  - step_size * q_prev
        q_next = q_prev  + (step_size * p_prev)/m
        
        # Update p and q                   
        p_prev = p_next
        q_prev = q_next 
        print(q_prev, p_prev)
        plt.hold(True)

def modified_euler_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
        # Assumes mass is constant for all particles
    p_prev = init_p
    q_prev = init_q
    
    
    for i in range(steps):
        # Plot phase space
        plt.figure(2)
        plt.plot(q_prev,p_prev,'r')
        # Calculate next p and q values
        p_next = p_prev  - step_size * q_prev
        q_next = q_prev  + (step_size * p_next)/m
        
        # Update p and q                   
        p_prev = p_next
        q_prev = q_next 
        print(q_prev, p_prev)
        plt.hold(True)

def leap_frog_method(init_q = 0, init_p = 1, step_size = 0.3, steps = 20, m = 1):
            # Assumes mass is constant for all particles
    p_prev = init_p
    q_prev = init_q
    
    for i in range(steps):
        # Plot phase space
        plt.figure(3)
        plt.plot(q_prev,p_prev,'r*')
        
        # leapfrog step added
        p_leapfrog = p_prev - (step_size/2)*q_prev
        q_next     = q_prev + (step_size*p_leapfrog)/m
        p_next     = p_leapfrog - (step_size/2)*q_next
        
        # Update p and q                   
        p_prev = p_next
        q_prev = q_next 
        print(q_prev, p_prev)
        plt.hold(True)     
                                 