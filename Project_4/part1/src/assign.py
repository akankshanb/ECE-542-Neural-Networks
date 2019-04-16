#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: assgin.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def assign_weight_count_all_0_case_1(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence

    Input node only receives digit '0' and all the gates are
    always open.

    Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
    """
    param_dict = {}
    param_dict['wgx'] = [[100.] if i == 0 else [0.] for i in range(10)]
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))

    param_dict['wix'] = np.zeros((in_dim, out_dim))
    param_dict['wih'] = np.zeros((out_dim, out_dim))
    param_dict['bi'] =  100. * np.ones((1, out_dim))

    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))

    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))

    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

def assign_weight_count_all_case_2(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence
        
        Input node only receives digit '0' and all the gates are
        always open.
        
        Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
        """
    param_dict = {}
    wgx1 =np.array([[100.] if i == 0 else [0.] for i in range(10)])
    wgx2 =np.array([[100.] if i == 2 else [0.] for i in range(10)])
    param_dict['wgx'] = list(np.concatenate((wgx1,wgx2),axis=1))
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))
    
    param_dict['wix'] = [[100.,100.] if i == 2 else [-100.,-100.] for i in range(10)]
    param_dict['wih'] = [[200,200],[200,200]]
    param_dict['bi'] =  np.zeros((1, out_dim))
    
    param_dict['wfx'] = np.zeros((in_dim, out_dim))
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))
    
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))
    
    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])

def assign_weight_count_all_case_3(cell, in_dim, out_dim):
    """ Parameters for counting all the '0' in the squence
        
        Input node receives all the digits '0' but input gate only
        opens for digit '0'. Other gates are always open.
        
        Args:
        in_dim (int): dimension of input
        out_dim (int): dimension of internal state and output
        """
    param_dict = {}
    wgx1 =np.array([[100.] if i == 0 else [0.] for i in range(10)])
    wgx2 =np.array([[100.] if i == 2 else [0.] for i in range(10)])
    param_dict['wgx'] = list(np.concatenate((wgx1,wgx2),axis=1))
    param_dict['wgh'] = np.zeros((out_dim, out_dim))
    param_dict['bg'] = np.zeros((1, out_dim))
    
    param_dict['wix'] = [[100.,100.] if i == 2 else [-100.,-100.] for i in range(10)]
    param_dict['wih'] = [[200,200],[200,200]]
    param_dict['bi'] =  np.zeros((1, out_dim))
    
    param_dict['wfx'] = [[-200.,-200.] if i == 3 else [0.,0.] for i in range(10)]
    param_dict['wfh'] = np.zeros((out_dim, out_dim))
    param_dict['bf'] = 100. * np.ones((1, out_dim))
    
    param_dict['wox'] = np.zeros((in_dim, out_dim))
    param_dict['woh'] = np.zeros((out_dim, out_dim))
    param_dict['bo'] = 100. * np.ones((1, out_dim))
    
    for key in param_dict:
        cell.set_config_by_name(key, param_dict[key])
