# -*- coding: utf-8 -*-
'''
svakulenko
27 December 2017

Pass tweets through the classifier and retweet
'''
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T

from tweepy.streaming import StreamListener
from tweepy import Stream, API, OAuthHandler

from inference import classify, load_params
from settings import *
from train import prepare_data
from twitter_settings import *

# set up Twitter connection
auth_handler = OAuthHandler(APP_KEY, APP_SECRET)
auth_handler.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
twitter_client = API(auth_handler)


class TweetClassifier(StreamListener):
    '''
    Overrides Tweepy class for Twitter Streaming API
    '''

    def __init__(self, model_path=MODEL_PATH, model_name='best_model.npz'):
        self.load_pretrained_model(model_path)

    def load_pretrained_model(self, model_path, model_name):
        # Load model and dictionaries
        print("Loading model params...")
        params = load_params('%s/%s' % (model_path, model_name))
        print("Loading dictionaries...")
        with open('%s/dict.pkl' % model_path, 'rb') as f:
            self.chardict = pkl.load(f)
        with open('%s/label_dict.pkl' % model_path, 'rb') as f:
            labeldict = pkl.load(f)

        self.n_char = len(self.chardict.keys()) + 1
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
        predictions = classify(tweet, t_mask, params, n_classes, self.n_char)
        # Theano function
        print("Compiling theano functions...")
        self.predict = theano.function([tweet,t_mask], predictions)

    def classify(self, X):
        x, x_m = prepare_data(X, self.chardict, n_chars=self.n_char)
        vp = self.predict(x, x_m)
        print vp
        ranks = np.argsort(vp)[:, ::-1]
        preds = []
        for idx, item in enumerate(X):
            preds.append(ranks[idx,:])
        return [ranks[0] for ranks in preds][0]

    def on_status(self, status):
        # ignore retweets
        if not hasattr(status,'retweeted_status') and status.in_reply_to_status_id == None:
            tweet_text = status.text
            tweet_id = status.id
            print(tweet_text)
            # retweet
            # twitter_client.retweet(id=tweet_id)

    def on_error(self, status_code):
        print (status_code, 'error code')


def stream_tweets():
    '''
    Connect to Twitter API and fetch relevant tweets from the stream
    '''
    listener = TweetClassifier()
    # start streaming
    while True:
        try:
            stream = Stream(auth_handler, listener)
            print ('Listening...')
            # stream.filter(track=["#nlpproc"])
            stream.sample()
        except Exception as e:
            # reconnect on exceptions
            print (e)
            continue


def test_classifier():
    classifier = TweetClassifier(model_name='best_model_81.npz')
    assert classifier.classify(["hot"]) == 1


if __name__ == '__main__':
    stream_tweets()
