# -*- coding: utf-8 -*-
'''
svakulenko
2 December 2017

Character-level neural network architecture based on Tweet2Vec implementation by bdhingra
https://github.com/bdhingra/tweet2vec
'''
from collections import OrderedDict

import numpy as np
import theano

from settings import *


def init_params(n_chars):
    '''
    Initialize all params
    '''
    params = OrderedDict()

    np.random.seed(0)

    # lookup table
    params['Wc'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(n_chars, IDIM)).astype('float32'), name='Wc')

    # f-GRU
    params['W_c2w_f_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(IDIM, HDIM)).astype('float32'), name='W_c2w_f_r')
    params['W_c2w_f_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(IDIM, HDIM)).astype('float32'), name='W_c2w_f_z')
    params['W_c2w_f_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(IDIM, HDIM)).astype('float32'), name='W_c2w_f_h')
    params['b_c2w_f_r'] = theano.shared(np.zeros((HDIM)).astype('float32'), name='b_c2w_f_r')
    params['b_c2w_f_z'] = theano.shared(np.zeros((HDIM)).astype('float32'), name='b_c2w_f_z')
    params['b_c2w_f_h'] = theano.shared(np.zeros((HDIM)).astype('float32'), name='b_c2w_f_h')
    params['U_c2w_f_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, HDIM)).astype('float32'), name='U_c2w_f_r')
    params['U_c2w_f_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, HDIM)).astype('float32'), name='U_c2w_f_z')
    params['U_c2w_f_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, HDIM)).astype('float32'), name='U_c2w_f_h')
    params['hid_ini_f'] = theano.shared(np.zeros((1, HDIM)).astype('float32'), name='hid_ini_f')

    # b-GRU
    params['W_c2w_b_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(IDIM, HDIM)).astype('float32'), name='W_c2w_b_r')
    params['W_c2w_b_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(IDIM, HDIM)).astype('float32'), name='W_c2w_b_z')
    params['W_c2w_b_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(IDIM, HDIM)).astype('float32'), name='W_c2w_b_h')
    params['b_c2w_b_r'] = theano.shared(np.zeros((HDIM)).astype('float32'), name='b_c2w_b_r')
    params['b_c2w_b_z'] = theano.shared(np.zeros((HDIM)).astype('float32'), name='b_c2w_b_z')
    params['b_c2w_b_h'] = theano.shared(np.zeros((HDIM)).astype('float32'), name='b_c2w_b_h')
    params['U_c2w_b_r'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, HDIM)).astype('float32'), name='U_c2w_b_r')
    params['U_c2w_b_z'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, HDIM)).astype('float32'), name='U_c2w_b_z')
    params['U_c2w_b_h'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, HDIM)).astype('float32'), name='U_c2w_b_h')
    params['hid_ini_b'] = theano.shared(np.zeros((1, HDIM)).astype('float32'), name='hid_ini_b')

    # dense
    params['W_c2w_df'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, LDIM)).astype('float32'), name='W_c2w_df')
    params['W_c2w_db'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(HDIM, LDIM)).astype('float32'), name='W_c2w_db')
    if BIAS:
        params['b_c2w_df'] = theano.shared(np.zeros((LDIM)).astype('float32'), name='b_c2w_db')
        params['b_c2w_db'] = theano.shared(np.zeros((LDIM)).astype('float32'), name='b_c2w_df')

    return params