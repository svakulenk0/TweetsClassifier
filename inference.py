# -*- coding: utf-8 -*-
'''
svakulenko
11 December 2017

Run inference on the pretrained model based on Tweet2Vec implementation by bdhingra: test_char.py
https://github.com/bdhingra/tweet2vec
'''
from collections import OrderedDict
import cPickle as pkl

import numpy as np
import lasagne
import theano
import theano.tensor as T

from settings import *
from train import BatchTweets, prepare_data
from tweet2vec import tweet2vec
from evaluate import precision


def load_params(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = vv

    return params


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense)


def infer(Xt, yt, model_path=MODEL_PATH):
    # Load model
    print("Loading model params...")
    params = load_params('%s/best_model.npz' % model_path)
    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)

    n_char = len(chardict.keys()) + 1
    n_classes = len(labeldict.keys())
    print "#classes:", n_classes
    print labeldict

    print("Building network...")
    
    # Tweet variables
    tweet = T.itensor3()
    targets = T.imatrix()

    # masks
    t_mask = T.fmatrix()

    # network for prediction
    predictions = classify(tweet, t_mask, params, n_classes, n_char)

    # Theano function
    print("Compiling theano functions...")
    predict = theano.function([tweet,t_mask], predictions)

    # Test
    print("Testing...")
    preds = []
    targs = []

    # iterator over batches
    xr, y = list(BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH))[0]
    print xr, y

    x, x_m = prepare_data(xr, chardict, n_chars=n_char)
    vp = predict(x, x_m)
    ranks = np.argsort(vp)[:, ::-1]

    for idx, item in enumerate(xr):
        preds.append(ranks[idx,:])
        targs.append(y[idx])

    print [ranks[0] for ranks in preds]
    # compute precision @1
    validation_cost = precision(np.asarray(preds), targs, 1)
    print validation_cost
    # print [labeldict.keys()[rank[0]] for rank in ranks]


def test_infer():
    # test generalization performance of the model
    X = ["hot", "hot and ", "ho", "cold"]
    y = ["hot", "hot", "hot", "cold"]
    infer(X, y)


if __name__ == '__main__':
    test_infer()
