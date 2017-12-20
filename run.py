# -*- coding: utf-8 -*-
'''
svakulenko
5 December 2017

Load tweets from MongoDB and train Tweet2Vec NN model
'''
from load_from_mongo import load_data_from_mongo_balanced
from train import split_dataset, train_model
from inference import test_model


def get_labeled_data():
    X, y = load_labeled_data_from_mongo_balanced("communityTweets", "cs_conferences",
                                         x_field="clean_text", y_field="label", limit=2000)
    assert X
    assert y
    print len(X), 'samples loaded'

    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
    train_model(X_train, y_train, X_validate, y_validate)
    test_model(X_test, y_test)


def test_get_cs_tweets():
    X, y = load_data_from_mongo_balanced("communityTweets", "cs_conferences",
                                         x_field="clean_text", y_value="CS", limit=30000)
    assert X
    assert y
    print len(X), 'samples loaded'
    print X[0]
    print y[0]


def test_get_random_tweets():
    X, y = load_data_from_mongo_balanced("tweets", "sample_04_12_2017",
                                         x_field="clean_text", y_value="random", limit=30000)
    assert X
    assert y
    print len(X), 'samples loaded'
    print X[0]
    print y[0]


def train_CS_news_recommender():
    # load CS tweets
    X1, y1 = load_data_from_mongo_balanced("communityTweets", "cs_conferences",
                                         x_field="clean_text", y_value="CS", limit=30000)
    assert X1
    assert y1
    print len(X1), 'CS samples loaded'

    # load random tweets
    X2, y2 = load_data_from_mongo_balanced("tweets", "sample_04_12_2017",
                                         x_field="clean_text", y_value="random", limit=30000)
    assert X2
    assert y2
    print len(X2), 'random samples loaded'

    # merge data
    X = X1 + X2
    y = y1 + y2
    print len(X), 'samples total'

    # split data
    assert len(X) == len(y)
    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_dataset(X, y)
    
    # run Tweet2Vec NN
    train_model(X_train, y_train, X_validate, y_validate)
    test_model(X_test, y_test)


if __name__ == '__main__':
    train_CS_news_recommender()
