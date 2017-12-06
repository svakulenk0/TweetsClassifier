# -*- coding: utf-8 -*-
'''
svakulenko
6 December 2017

All networks parameters are stored here
based on Tweet2Vec implementation by bdhingra https://github.com/bdhingra/tweet2vec
'''

# network size
CHAR_DIM = 6  # number of unique characters in the input layer
HDIM = 500  # size of the hidden layer
LDIM = 1  # number of the unique output labels (categories)
MAX_LENGTH = 4  # max sequence of characters length for the input layer
# Twitter limits 140/280

# training parameters
SCALE = 0.1  # Initialization scale
BIAS = False  # use bias
NUM_EPOCHS = 30 # Number of epochs
N_BATCH = 64  # Batch size
LEARNING_RATE = 0.01
MOMENTUM = 0.9
REGULARIZATION = 0.0001
SCHEDULE = True  # use schedule
GRAD_CLIP = 5.  # gradient clipping for regularization

# logging and model back ups
DISPF = 5  # Display frequency
SAVEF = 1000  # Save frequency
