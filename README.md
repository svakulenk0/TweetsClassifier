# Tweet Classifier


## Motivation

Train a classifier to recommend relevant tweets.


## Approach

* character-level neural network with bi-GRU architecture based on [Tweet2Vec implementation](https://github.com/bdhingra/tweet2vec)


## Run

* Short test: THEANO_FLAGS='floatX=float32' python train.py

* Train: THEANO_FLAGS='floatX=float32' python run.py


## Requirements

* pymongo


## Datasets

* tweets with the Computer Science conference hashtags
* [ArchiveTeam JSON Download of Twitter Stream 2017-02](https://archive.org/details/archiveteam-twitter-stream-2017-02)


## Evaluation results

It is important to provide input samples equally balanced between all the classes otherwised the results are skewed towards the most frequent classes.

1. CS topics

Dataset: 5 classes * 2,000 tweets each = 10,000 tweets in total
Random guess: 0.2 uniform probability distribution
Model: 0.234375

2. ML vs NLP ?

Dataset: 2 classes * 5,000 tweets each = 10,000 tweets in total
Random guess: 0.5 uniform probability distribution
Model:

3. CS vs random tweets

Dataset: 2 classes * 30,000 tweets each = 60,000 tweets in total
Random guess: 0.5 uniform probability distribution
# characters: 128

* Run 1

split: 0.8 x 0.1 x 0.1

learning rate: 0.3

Epoch 12 Training Cost 0.00590342712402 Validation Precision 0.776462743791 Regularization Cost 3.82598352432 Max Precision 0.859375

Test: 0.6875

* Run 2

split: 0.6 x 0.2 x 0.2

learning rate: 0.3

Epoch 11 Training Cost 0.00385822719998 Validation Precision 0.742083333333 Regularization Cost 3.74113941193 Max Precision 0.8125

Test: 0.8125

* Run 3

split: 0.6 x 0.2 x 0.2

learning rate: 0.1


## References

* [Tweet2Vec](https://github.com/bdhingra/tweet2vec)
