# -*- coding: utf-8 -*-
'''
svakulenko
5 December 2017

Load tweets from MongoDB and train Tweet2Vec NN model
'''
from load_from_mongo import load_data_from_mongo, NTWEETS
from train import split_dataset, train_model
from inference import test_model


X, y = load_data_from_mongo("communityTweets", "cs_conferences",
                            x_field="clean_text", y_field="label", limit=NTWEETS)
assert X
assert y
print len(X), 'samples loaded'

(X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
train_model(X_train, y_train, X_validate, y_validate)
test_model(X_test, y_test)
