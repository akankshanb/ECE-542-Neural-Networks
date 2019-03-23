#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action = 'store_true',
                        help = 'Test the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 20, 10])
    #model = network2.load('saver')
    #train the network using SGD
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=0.5e-3,
        lmbda = 0.0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    training_accuracy = np.divide(training_accuracy, len(train_data[0]))*100
    evaluation_accuracy = np.divide(evaluation_accuracy, len(test_data[0]))*100
    fig1, ax1 = plt.subplots()
    ax1.plot(training_cost, 'k', label = 'Training cost')
    ax1.plot(evaluation_cost, 'r', label = 'Validation cost')
    ax1.legend()
    plt.title('Training cost vs Evaluation cost')
    legend1 = ax1.legend(loc='upper center', fontsize='medium')
    plt.ylabel('Cost')
    plt.xlabel('Number of epochs')
    plt.savefig('Cost')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(training_accuracy, 'k', label = 'Training accuracy')
    ax2.plot(evaluation_accuracy, 'r', label = 'Validation accuracy')
    ax2.legend()
    plt.title('Training accuracy vs Evaluation accuracy')
    legend2 = ax2.legend(loc='upper left', fontsize='medium')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Number of epochs')
    plt.savefig('Accuracy')
    plt.show()
    model.save('saver')

def main_test():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    #model = network2.Network([784, 20, 10])
    model = network2.load('saver')
    #test the network using SGD
    test_cost = model.total_cost(test_data, lmbda = 0.0, convert=True)
    test_accuracy = model.accuracy(test_data, convert=False)
    
    print('Test cost : ', test_cost)
    print('Test Accuracy : ', test_accuracy)


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        main_test()
