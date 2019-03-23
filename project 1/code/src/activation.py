#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    sig = 1/(1+np.exp(-z))
    return sig

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    sigDerivative = np.exp(z)/((1+np.exp(z))**2)
    return sigDerivative
