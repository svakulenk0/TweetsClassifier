# Tweet Classifier


## Motivation

Train a classifier to recommend relevant tweets.


## Aproach

* character-level neural network is based on [Tweet2Vec implementation](https://github.com/bdhingra/tweet2vec)


## Run

* Test: THEANO_FLAGS='floatX=float32' python train.py


## Requirements

* pymongo


## Datasets

* [ArchiveTeam JSON Download of Twitter Stream 2017-02](https://archive.org/details/archiveteam-twitter-stream-2017-02)

## References

* [Tweet2Vec](https://github.com/bdhingra/tweet2vec)
