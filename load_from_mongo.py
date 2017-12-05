# -*- coding: utf-8 -*-
'''
svakulenko
2 December 2017

Loading tweets from MongoDB
'''
import string

from pymongo import MongoClient


# mapping keywords to topic labels
LABELS = {'AI': ['aaai17', 'aaai2017', 'ijcai2017'],
          'NLP': ['naacl2016', 'acl2016berlin', 'acl2016', 'emnlp2016', 'lrec2016', 'eacl2017', 'acl2017', 'ijcnlp2017taipei', 'nlproc'],
          'IR': ['sigir2016', 'recsys2016', 'ictir2016', 'sigir2017', 'ecir2017', 'ecir2016'],
          'SemanticWeb': ['iswc2016', 'iswc2017', 'eswc2016'],
          'WebScience': ['websci16', 'webscience16', 'wsdm2017', 'kdd2016','www2016ca', 'kddnews', 'cikm2016', 'www2017perth', 'www2017', 'icwe2017', 'cikm2017', 'icwsm', 'cultureanalytics2016'],
          'ML': ['nipsconference', 'nips', 'nips2016', 'nips2017', 'jmlr', 'icml2016', 'wiml2016', 'iclr2016', 'iclr2017', 'iclr', 'reworkdl']
         }
NTWEETS = 50000

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


def load_data_from_mongo(db, collection, x_field, y_field, limit):
    X = []
    y = []
    collection = connect_to_mongo(db, collection)
    for doc in collection.find({y_field: {'$exists': True}}, limit=limit):
        X.append(doc[x_field])
        y.append(doc[y_field])
        # print(doc["text"])
    return (X, y)


def detect_keywords(tokens, labels):
    topic = None
    for label, keywords in labels.items():
        for index, token in enumerate(tokens):
            if token.lower().strip('#') in keywords:
                # save topic label (assume there is only one label per tweet)
                topic = label
                # remove token
                del tokens[index]
    return topic, " ".join(tokens)


def test_detect_keywords():
    # the original text of the tweet post
    tweet = "@sigir2016 is a great place to be"
    # remove punctuation
    tweet = tweet.encode('utf-8').translate(None, string.punctuation)
    tokens = tweet.split()

    assert detect_keywords(tokens, LABELS) == 'IR'
    print detect_keywords(tokens, LABELS)


def label_tweets(db, collection, labels, limit):
    # print labels.values
    collection = connect_to_mongo(db, collection)
    # iterate over the tweets
    for doc in collection.find(limit=limit):
        # the original text of the tweet post
        tweet = doc["text"]
        # remove punctuation
        tweet = tweet.encode('utf-8').translate(None, string.punctuation)
        tokens = tweet.split()

        label, clean_text = detect_keywords(tokens, labels)
        # save label and cleaned text string into MongoDB
        collection.update({"_id": doc["_id"]}, {"$set": {"label": label,
                                                         "clean_text": clean_text}})
        # if not label:
        #     print(tokens)


def test_label_tweets():
    label_tweets("communityTweets", "cs_conferences", LABELS, NTWEETS)


def test_load_data_from_mongo():
    X, y = load_data_from_mongo("communityTweets", "cs_conferences",
                                x_field="clean_text", y_field="label", limit=2)
    assert X
    print len(X), 'samples loaded'
    print X
    print y


def test_count_tweets():
    count_tweets("communityTweets", "cs_conferences")


def test_connect_to_mongo():
    posts = connect_to_mongo("test", "test_connect_to_mongo")
    # insert sample doc
    posts.insert_one({"test": "test"})
    assert posts.count()
    print posts.count()


if __name__ == '__main__':
    # test_detect_keywords()
    # test_label_tweets()
    # test_count_tweets()
    test_load_data_from_mongo()
