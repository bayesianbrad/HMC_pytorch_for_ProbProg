#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:18:19 2017

@author: bradley
"""
# Using super() method
# Understanding the Methord Resolution Order(MRO)
# How to deal with unknown number of arguments
# by using defensive programming and having a 
# root class 'Root' absorb any additional 
# *kwrds that havent been assigned
class Root:
    def draw(self):
        # the delegation chain stops here
        assert not hasattr(super(), 'draw')

class Shape(Root):
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting shape to:', self.shapename)
        super().draw()

class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting color to:', self.color)
        super().draw()

cs = ColoredShape(color='blue', shapename='square')
cs.draw()

#How to Incorporate a Non-cooperative Class
# Occasionally, a subclass may want to use 
#cooperative multiple inheritance techniques 
#with a third-party class that wasn’t 
#designed for it (perhaps its method of interest 
#doesn’t use super() or perhaps the class
#doesn’t inherit from the root class).
#This situation is easily remedied by creating 
#an adapter class that plays by the rules.
#
#For example, the following Moveable class does not make 
#super() calls, and it has an __init__() signature 
#that is incompatible with object.__init__, and it 
#does not inherit from Root:

class Moveable:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def draw(self):
        print('Drawing at position:', self.x, self.y)
#If we want to use this class with our cooperatively 
#designed ColoredShape hierarchy, we need to make an 
#adapter with the requisite super() calls:

class MoveableAdapter(Root):
    def __init__(self, x, y, **kwds):
        self.movable = Moveable(x, y)
        super().__init__(**kwds)
    def draw(self):
        self.movable.draw()
        super().draw()

class MovableColoredShape(ColoredShape, MoveableAdapter):
    pass

MovableColoredShape(color='red', shapename='triangle',
                    x=10, y=20).draw()