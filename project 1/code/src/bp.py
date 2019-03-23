#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]
    
    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    a = [x]
    activations = [x]

    # print('No of layers ', num_layers)
    # print('Shape of x ', x.shape)
    # print('Shape of y ', y.shape)
    # print('Shape of weightsT[0] ', weightsT[0].shape)
    # print('Shape of weightsT[1] ', weightsT[1].shape)
    # print('Shape of biases[0] ', biases[0].shape)
    # print('Shape of biases[1] ', biases[1].shape)
    # print('Shape of a[0] ', a[0].shape)
    # print('Shape of activations[0] ', activations[0].shape)

    for i in range(num_layers-1):
    	a.append(np.dot(weightsT[i], activations[-1])+biases[i])
    	activations.append(sigmoid(a[-1]))
    ###
    
    # print('Shape of a[0] ', a[0].shape)
    # print('Shape of a[1] ', a[1].shape)
    # print('Shape of a[2] ', a[2].shape)
    # print('Shape of activations[0] ', activations[0].shape)
    # print('Shape of activations[1] ', activations[1].shape)
    # print('Shape of activations[2] ', activations[2].shape)

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)

    # print('Shape of G ', delta.shape)
    # print('Shape of nabla_b[0] ', nabla_b[0].shape)
    # print('Shape of nabla_wT[0] ', nabla_wT[0].shape)
    # print('Shape of nabla_b[1] ', nabla_b[1].shape)
    # print('Shape of nabla_wT[1] ', nabla_wT[1].shape)
    # print('\n Backprop \n')

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    G = delta

    for k in range(num_layers-1, 0, -1):
    	
    	nabla_b[k-1] = G
    	nabla_wT[k-1] = np.dot(G, np.transpose(activations[k-1]))
    	G = np.dot(np.transpose(weightsT[k-1]), G)
    	G = np.multiply(G, sigmoid_prime(a[k-1]))

    	# print('Shape of nabla_b[', k-1, '] ', nabla_b[k-1].shape)
    	# print('Shape of nabla_wT[', k-1, '] ', nabla_wT[k-1].shape)

    # print('\n\n')
    return (nabla_b, nabla_wT)

