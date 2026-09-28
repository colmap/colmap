""" generic functions
"""
import pdb
import numpy as np


def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(a * (b - x)))


def sigmoid_range(x, at5, at95):
    """ create sigmoid function like that: 
            sigmoid(at5)  = 0.05
            sigmoid(at95) = 0.95
        and returns sigmoid(x) 
    """
    a = 6 / (at95 - at5)
    b = at5 + 3 / a
    return sigmoid(x, a, b)
