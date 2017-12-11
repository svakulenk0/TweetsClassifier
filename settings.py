# -*- coding: utf-8 -*-
'''
svakulenko
6 December 2017

All networks parameters are stored here
based on Tweet2Vec implementation by bdhingra https://github.com/bdhingra/tweet2vec
'''

# network size
CHAR_DIM = 150  # dimensionality of the character embeddings lookup
# 128 US-ASCII + 1,920 UTF-8
HDIM = 500  # size of the hidden layer
LDIM = 5  # number of the unique output labels (categories)
MAX_LENGTH = 280  # max sequence of characters length for the input layer
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

# thresholds
T1 = 0.01  # in learning schedule: if change < T1
T2 = 0.0001  # stopping criterion: if sum(deltas) / len(deltas) < T2

# logging and model back ups
DISPF = 5  # Display frequency
SAVEF = 1000  # Save frequency
MODEL_PATH = "./model"
