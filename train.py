# -*- coding: utf-8 -*-
'''
svakulenko
2 December 2017

Training a character-level NN classifier based on Tweet2Vec implementation by bdhingra
https://github.com/bdhingra/tweet2vec
'''
from collections import Counter
import time
import cPickle as pkl

import numpy as np
from sklearn.model_selection import train_test_split
import lasagne
import theano
import theano.tensor as T

from tweet2vec import init_params, tweet2vec
from settings import *


def build_dictionary(text):
    '''
    Build the character dictionary
    text: list of tweets
    '''
    charcount = Counter()
    for cc in text:
        chars = list(cc)
        for c in chars:
            charcount[c] += 1
    chars = charcount.keys()
    freqs = charcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    chardict = Counter()
    for idx, sidx in enumerate(sorted_idx):
        chardict[chars[sidx]] = idx + 1

    return chardict, charcount


def build_label_dictionary(labels):
    """
    Build the label dictionary
    labels: list of labels
    """
    labelcount = Counter()
    for l in labels:
        labelcount[l] += 1
    labels = labelcount.keys()
    freqs = labelcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    labeldict = Counter()
    for idx, sidx in enumerate(sorted_idx):
        labeldict[labels[sidx]] = idx + 1
    return labeldict, labelcount


def save_dictionary(worddict, wordcount, loc):
    '''
    Save a dictionary into the specified location 
    '''
    with open(loc, 'w') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


class BatchTweets():

    def __init__(self, data, targets, labeldict, batch_size=128, max_classes=1000, test=False):
        # convert targets to indices
        if not test:
            tags = []
            for l in targets:
                tags.append(labeldict[l] if l in labeldict and labeldict[l] < max_classes else 0)
        else:
            tags = []
            for line in targets:
                tags.append([labeldict[l] if l in labeldict and labeldict[l] < max_classes else 0 for l in line])

        self.batch_size = batch_size
        self.data = data
        self.targets = tags

        self.prepare()
        self.reset()

    def prepare(self):
        self.indices = np.arange(len(self.data))
        self.curr_indices = np.random.permutation(self.indices)

    def reset(self):
        self.curr_indices = np.random.permutation(self.indices)
        self.curr_pos = 0
        self.curr_remaining = len(self.curr_indices)

    def next(self):
        if self.curr_pos >= len(self.indices):
            self.reset()
            raise StopIteration()

        # current batch size
        curr_batch_size = np.minimum(self.batch_size, self.curr_remaining)

        # indices for current batch
        curr_indices = self.curr_indices[self.curr_pos:self.curr_pos+curr_batch_size]
        self.curr_pos += curr_batch_size
        self.curr_remaining -= curr_batch_size

        # data and targets for current batch
        x = [self.data[ii] for ii in curr_indices]
        y = [self.targets[ii] for ii in curr_indices]

        return x, y

    def __iter__(self):
        return self


def schedule(lr):
    print("Updating learning rate...")
    lr = max(1e-5, lr/2)
    return lr


def prepare_data(seqs_x, chardict, n_chars=1000):
    """
    Prepare the data for training - add masks
    """
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)
    x = np.zeros((n_samples, MAX_LENGTH)).astype('int32')
    x_mask = np.zeros((n_samples, MAX_LENGTH)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.

    return np.expand_dims(x, axis=2), x_mask


def split_dataset(X, y, train_size=0.6, test_size=0.2):
    '''
    Split the dataset into training, validation and test sets evenly distributing the class labels samples
    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    '''
    hold_size = 1 - train_size
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=hold_size, random_state=42, stratify=y)
    X_validate, X_test, y_validate, y_test = train_test_split(X_hold, y_hold, test_size=test_size/hold_size, random_state=42, stratify=y_hold)
    return [(X_train, y_train), (X_validate, y_validate), (X_test, y_test)]


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), l_dense, lasagne.layers.get_output(emb_layer)


def train(Xt, yt, Xv, yv, save_path,
          num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, mu=MOMENTUM,
          reg=REGULARIZATION, sch=SCHEDULE):
    '''
    Xt, yt        X,y arrays with training data
    Xv, yv        X,y arrays with validation data split
    save_path     path to store the trained model
    '''

    print("Initializing model...")
    
    # Build dictionaries from training data
    chardict, charcount = build_dictionary(Xt)
    n_char = len(chardict.keys()) + 1
    save_dictionary(chardict, charcount, '%s/dict.pkl' % save_path)
    
    # params
    params = init_params(n_chars=n_char)
    
    labeldict, labelcount = build_label_dictionary(yt)
    save_dictionary(labeldict, labelcount, '%s/label_dict.pkl' % save_path)

    n_classes = len(labeldict.keys())

    # classification params
    params['W_cl'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(LDIM, n_classes)).astype('float32'), name='W_cl')
    params['b_cl'] = theano.shared(np.zeros((n_classes)).astype('float32'), name='b_cl')

    # iterators
    train_iter = BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=n_classes)
    val_iter = BatchTweets(Xv, yv, labeldict, batch_size=N_BATCH, max_classes=n_classes, test=True)

    print("Building...")
    
    # Tweet variables
    tweet = T.itensor3()
    targets = T.ivector()
    
    # masks
    t_mask = T.fmatrix()

    # network for prediction
    predictions, net, emb = classify(tweet, t_mask, params, n_classes, n_char)

    # batch loss
    loss = lasagne.objectives.categorical_crossentropy(predictions, targets)
    cost = T.mean(loss) + reg * lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)
    cost_only = T.mean(loss)
    reg_only = reg * lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)

    # updates
    print("Computing updates...")
    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)

    # Theano functions
    print("Compiling theano functions...")
    inps = [tweet, t_mask, targets]
    predict = theano.function([tweet, t_mask], predictions)
    cost_val = theano.function(inps, [cost_only, emb])
    train = theano.function(inps, cost, updates=updates)
    reg_val = theano.function([], reg_only)

    # Training
    print("Training...")
    uidx = 0
    maxp = 0.
    start = time.time()
    valcosts = []
    for epoch in range(num_epochs):
        n_samples = 0
        train_cost = 0.
        print("Epoch {}".format(epoch))

        # learning schedule
        if len(valcosts) > 1 and sch:
            change = (valcosts[-1] - valcosts[-2]) / abs(valcosts[-2])
            if change < T1:
                lr = schedule(lr)
                updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)
                train = theano.function(inps, cost, updates=updates)
                T1 = T1/2

        # stopping criterion
        if len(valcosts) > 6:
            deltas = []
            for i in range(5):
                deltas.append((valcosts[-i-1] - valcosts[-i-2]) / abs(valcosts[-i-2]))
            if sum(deltas) / len(deltas) < T2:
                break

        ud_start = time.time()
        for xr,y in train_iter:
            n_samples +=len(xr)
            uidx += 1
            x, x_m = prepare_data(xr, chardict, n_chars=n_char)
            if x is None:
                print("Minibatch with zero samples under maxlength.")
                uidx -= 1
                continue

        curr_cost = train(x,x_m,y)
        train_cost += curr_cost*len(xr)
        ud = time.time() - ud_start

        if np.isnan(curr_cost) or np.isinf(curr_cost):
            print("Nan detected.")
            return

        if np.mod(uidx, DISPF) == 0:
            print("Epoch {} Update {} Cost {} Time {}".format(epoch,uidx,curr_cost,ud))

        if np.mod(uidx,SAVEF) == 0:
            print("Saving...")
        
        saveparams = OrderedDict()
        for kk,vv in params.iteritems():
            saveparams[kk] = vv.get_value()
            np.savez('%s/model.npz' % save_path,**saveparams)
        
        print("Done.")

        print("Testing on Validation set...")
        
        preds = []
        targs = []
        
        for xr,y in val_iter:
            x, x_m = prepare_data(xr, chardict, n_chars=n_char)
            if x is None:
                print("Validation: Minibatch with zero samples under maxlength.")
                continue

            vp = predict(x,x_m)
            ranks = np.argsort(vp)[:,::-1]
            for idx,item in enumerate(xr):
                preds.append(ranks[idx,:])
                targs.append(y[idx])

            validation_cost = precision(np.asarray(preds),targs,1)
            regularization_cost = reg_val()

            if validation_cost > maxp:
                maxp = validation_cost
                saveparams = OrderedDict()
                for kk,vv in params.iteritems():
                    saveparams[kk] = vv.get_value()
                np.savez('%s/best_model.npz' % (save_path),**saveparams)

        print("Epoch {} Training Cost {} Validation Precision {} Regularization Cost {} Max Precision {}".format(epoch, train_cost/n_samples, validation_cost, regularization_cost, maxp))
        print("Seen {} samples.".format(n_samples))
        valcosts.append(validation_cost)

        print("Saving...")
        saveparams = OrderedDict()
        for kk, vv in params.iteritems():
            saveparams[kk] = vv.get_value()
        np.savez('%s/model_%d.npz' % (save_path,epoch),**saveparams)
        print("Done.")

    print("Finish. Total training time = {}".format(time.time()-start))


def test_split_dataset():
    '''
    Generate sample dataset with 100 samples and split it into 3 sets
    '''
    X = [1, 2, 3, 4, 5, 2, 3, 4, 3, 5, 5, 2, 3, 4, 5]
    y = [True, False, True, True, True, False, True, False, True, False, True, False, True, False, True]
    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
    assert True in y_train and False in y_train
    assert True in y_validate and False in y_validate
    assert True in y_test and False in y_test


def test_train():
    '''
    Generate sample dataset and split it into 3 sets
    '''
    X = ["test" for i in range(20)]
    y = ["true" for i in range(20)]
    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
    train(X_train, y_train, X_validate, y_validate, save_path="./model")


if __name__ == '__main__':
    # test_split_dataset()
    test_train()
