from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MF(object):
    def __init__(self, Y, K, lam = 0.1, Xinit = None, Winit = None,
                learning_rate = 0.5, max_iter = 1000, print_every = 100):
        self.Y = Y                                      # represents the utility matrix
        self.K = K
        self.lam = lam                                  # regularization parameter
        self.learning_rate = learning_rate              # for gradient descent
        self.max_iter = max_iter                        # maximum number of iterations
        self.print_every = print_every                  # print loss after each a few iters
        self.n_users = int(np.max(Y[:, 0])) + 1
        self.n_items = int(np.max(Y[:, 1])) + 1
        self.n_ratings = Y.shape[0]                     # number of known ratings
        self.X = np.random.randn(self.n_items, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.n_users) if Winit is None else Winit
        self.b = np.random.randn(self.n_items)          # item biases
        self.d = np.random.randn(self.n_users)          # user biases
    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            # user_id, item_id, rating
            n       = int(self.Y[i, 0])
            m       = int(self.Y[i, 1])
            rating  = self.Y[i, 2]
            L += 0.5 * (self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating) ** 2

        L /= self.n_ratings
        # regularization, don’t ever forget this
        return L + 0.5 * self.lam * (np.sum(self.X ** 2) + np.sum(self.W ** 2))

    # def updateXb(self):
        



