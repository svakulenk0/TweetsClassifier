# -*- coding: utf-8 -*-
'''
svakulenko
31 March 2019

Pass tweets through the classifier and retweet
Loading the job offers classification model trained by Yoan Bachev
'''
import string
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T

from tweepy.streaming import StreamListener
from tweepy import Stream, API, OAuthHandler, Cursor

from inference import classify, load_params
from settings import *
from train import prepare_data
from twitter_settings import *
from load_from_mongo import clean_tokens

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAX_NB_WORDS = 5000 # consider up to x most occuring words in dataset


class TweetClassifier(StreamListener):
    '''
    Overrides Tweepy class for Twitter Streaming API
    '''

    def __init__(self, model_path=MODEL_PATH, model_name='x1.sav'):
        self.load_pretrained_model(model_path, model_name)
        # set up Twitter connection
        self.auth_handler = OAuthHandler(APP_KEY, APP_SECRET)
        self.auth_handler.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
        self.api = API(self.auth_handler)

    def load_pretrained_model(self, model_path, model_name):
        # Load model and dictionaries
        print("Loading pre-trained model...")
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        self.model = pickle.load(open('%s/%s' % (model_path, model_name), 'rb'))

    def on_status(self, status):
        # ignore retweets
        if not hasattr(status,'retweeted_status') and status.in_reply_to_status_id == None:
            tweet_text = status.text
            tweet_id = status.id
            print(tweet_text)

            # preprocess
            sequences= self.tokenizer.texts_to_sequences([tweet_text])
            dat = pad_sequences(sequences, maxlen=1000)

            # classify
            prediction = self.model.predict(dat, batch_size=64)[0,1]

            if prediction >0.4:
                print tweet_text
                print prediction
            #     # retweet
            #     self.api.update_status(status='https://twitter.com/%s/status/%s' % (tweet['user']['screen_name'], tweet['id']))
            # retweet
            # twitter_client.retweet(id=tweet_id)

    def on_error(self, status_code):
        print (status_code, 'error code')


def stream_tweets():
    '''
    Connect to Twitter API and fetch relevant tweets from the stream
    '''
    # get users from list
    listener = TweetClassifier()
    # members = [member.id_str for member in Cursor(listener.api.list_members, MY_NAME, LIST).items()]

    # start streaming
    while True:
        try:
            stream = Stream(listener.auth_handler, listener)
            print ('Listening...')
            # stream.filter(track=["#nlpproc"])
            # stream.filter(follow=members)
            stream.sample(languages=['en'])
        except Exception as e:
            # reconnect on exceptions
            print (e)
            continue


def test_classifier():
    classifier = TweetClassifier()
    # classifier = TweetClassifier(model_name='best_model_81.npz')
    assert classifier.classify(["hot"]) == 1


if __name__ == '__main__':
    stream_tweets()
