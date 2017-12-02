# -*- coding: utf-8 -*-
'''
svakulenko
2 December 2017

Loading tweets from MongoDB
'''

from pymongo import MongoClient


def connect_to_mongo(db, collection):
    client = MongoClient('localhost', 27017)
    return client[db][collection]


def count_tweets(db, collection):
    tweets = connect_to_mongo(db, collection)
    assert tweets.count()
    print tweets.count()


def test_cursor():
    collection = connect_to_mongo("communityTweets", "cs_conferences")
    # show one of the documents
    for doc in collection.find(limit=1):
        print(doc["text"])


def load_data_from_mongo(db, collection, x_field, limit):
    X = []
    y = []
    collection = connect_to_mongo(db, collection)
    for doc in collection.find(limit=limit):
        X.append(doc[x_field])
        # yt.append(yc)
        # print(doc["text"])


def test_load_data_from_mongo():
    X, y = load_data_from_mongo("communityTweets", "cs_conferences", x_field="text", limit=2)
    assert X
    print len(X), 'samples loaded'
    print X


def test_count_tweets():
    count_tweets("communityTweets", "cs_conferences")


def test_connect_to_mongo():
    posts = connect_to_mongo("test", "test_connect_to_mongo")
    # insert sample doc
    posts.insert_one({"test": "test"})
    assert posts.count()
    print posts.count()


if __name__ == '__main__':
    test_load_data_from_mongo()
