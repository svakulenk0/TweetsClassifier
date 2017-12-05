# -*- coding: utf-8 -*-
'''
svakulenko
5 December 2017

Load tweets from MongoDB and train Tweet2Vec NN model
'''
from load_from_mongo import load_data_from_mongo
from train import split_dataset, train


X, y = load_data_from_mongo("communityTweets", "cs_conferences",
                            x_field="clean_text", y_field="label", limit=20)
(X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
train(train_data, val_data, save_path="./model/")
