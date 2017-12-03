# -*- coding: utf-8 -*-
'''
svakulenko
2 December 2017

Loading tweets from MongoDB
'''

from pymongo import MongoClient


# mapping keywords to topic labels
LABELS = {'AI': ['aaai17', 'aaai2017', 'ijcai2017'],
          'NLP': ['naacl2016', 'acl2016berlin', '@acl2016', 'emnlp2016', 'lrec2016', 'eacl2017', 'acl2017', 'ijcnlp2017'],
          'IR': ['sigir2016', 'recsys2016', 'ictir2016', 'sigir2017', 'ecir2017', 'ecir2016'],
          'SemanticWeb': ['iswc2016', 'iswc2017', 'eswc'],
          'WebScience': ['websci16', 'wsdm2017', 'kdd2016', 'cikm2016', 'www2017perth', 'www2017', 'icwe2017', 'cikm2017'],
          'ML': ['nips2016', 'nips2017', 'jmlr', 'icml2016', 'wiml2016', 'iclr2016', 'iclr2017', 'iclr']
         }


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
    return (X, y)


def label_tweets(db, collection, labels, limit):
    # print labels.values
    collection = connect_to_mongo(db, collection)
    # show one of the documents
    for doc in collection.find(limit=limit):
        tweet = doc["text"].split()
        # print tweet
        for topic, keywords in labels.items():
            for word in tweet:
                if word.lower() in keywords:
                    # save topic label
                    pass
                    # print topic
                    # print(doc["text"])
                else:
                    print(doc["text"])


def test_label_tweets():
    label_tweets("communityTweets", "cs_conferences", LABELS, 10)


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
    test_label_tweets()
