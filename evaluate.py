# -*- coding: utf-8 -*-
'''
svakulenko
2 December 2017

Standard evaluation metrics for a classifier performance based on Tweet2Vec implementation by bdhingra
https://github.com/bdhingra/tweet2vec
'''

def precision(p, t, k):
    '''
    Compute precision @ k for predictions p and targets t
    '''
    n = p.shape[0]
    res = np.zeros(n)
    for idx in range(n):
        index = p[idx, :k]
        for i in index:
            if i in t[idx]:
                res[idx] += 1
    return np.sum(res) / (n * k)
