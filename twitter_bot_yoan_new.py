# -*- coding: utf-8 -*-
'''
svakulenko
6 May 2019

Pass tweets through the classifier and retweet
Loading the job offers classification model trained by Yoan Bachev
'''
import pickle

from twython import Twython
from twython import TwythonStreamer

from settings import *
from twitter_settings import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAX_NB_WORDS = 5000 # consider up to x most occuring words in dataset

# load pre-trained model
model_path='model.sav'
model = pickle.load(open('%s' % (model_path), 'rb'))

class MyStreamer(TwythonStreamer):
    '''
    Overrides Tweepy class for Twitter Streaming API
    '''

    # def __init__(self, model_path=MODEL_PATH, model_name='model.sav'):
    #     self.load_pretrained_model(model_path, model_name)
    #     # set up Twitter connection
    #     self.auth_handler = OAuthHandler(APP_KEY, APP_SECRET)
    #     self.auth_handler.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    #     self.api = API(self.auth_handler)

    def on_success(self, data):
        if 'text' in data:
            sequences= tokenizer.texts_to_sequences([data['text']])
            dat = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            prediction = model.predict(dat, batch_size=64)[0,1]

            if prediction > 0.8:
                print(data['text'])
                print(prediction)

        # ignore retweets
        # if not hasattr(status,'retweeted_status') and status.in_reply_to_status_id == None:
        #     tweet_text = status.text
        #     tweet_id = status.id
        #     # print(tweet_text)

        #     # preprocess
        #     sequences= self.tokenizer.texts_to_sequences([tweet_text])
        #     dat = pad_sequences(sequences, maxlen=1000)

        #     # classify
        #     prediction = self.model.predict(dat, batch_size=64)[0,1]

            #     # retweet
            #     self.api.update_status(status='https://twitter.com/%s/status/%s' % (tweet['user']['screen_name'], tweet['id']))
            # retweet
            # twitter_client.retweet(id=tweet_id)

    def on_error(self, status_code, data):
        print(status_code)


def stream_tweets():
    '''
    Connect to Twitter API and fetch relevant tweets from the stream
    '''
    # get users from list
    stream = MyStreamer(APP_KEY, APP_SECRET,
                        OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    # members = [member.id_str for member in Cursor(listener.api.list_members, MY_NAME, LIST).items()]
    stream.statuses.filter(track='Hirring , Job , CareerArc ',language='en')
    # start streaming
    # while True:
    #     try:
    #         stream = Stream(listener.auth_handler, listener)
    #         print ('Listening...')
    #         stream.filter(track=['hiring', 'job', 'career'], languages=['en'])
    #         # stream.filter(follow=members)
    #         # stream.sample(languages=['en'])
    #     except Exception as e:
    #         # reconnect on exceptions
    #         print (e)
    #         continue



if __name__ == '__main__':
    stream_tweets()
