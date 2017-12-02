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


def test_connect_to_mongo():
    posts = connect_to_mongo("test", "test")
    # insert sample doc
    posts.insert_one({"test": "test"})
    assert posts.count()
    print posts.count()


if __name__ == '__main__':
    test_connect_to_mongo()
