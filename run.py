# -*- coding: utf-8 -*-
'''
svakulenko
5 December 2017

Load tweets from MongoDB and train Tweet2Vec NN model
'''
from load_from_mongo import load_data_from_mongo, NTWEETS
from train import split_dataset, train


X, y = load_data_from_mongo("communityTweets", "cs_conferences",
                            x_field="clean_text", y_field="label", limit=NTWEETS)
assert X
assert y
print len(X), 'samples loaded'

(X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
train(X_train, y_train, X_validate, y_validate, save_path="./model")
