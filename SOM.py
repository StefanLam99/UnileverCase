# Simple implementation of the SOM network
# Author: Stefan Lam

import numpy as np
from Utils import euclidean_distance
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from matplotlib import patches
from time import time


class SOM:
    """
    Class to make a SOM network model
    """

    def __init__(self, X, map_shape = (8,8), init_lr=0.1, init_response=1, max_iter=10000, normalize_data=False, seed=0):
        np.random.seed(seed)  # set the seed for this SOM network

        # input data and output map
        self.X = X
        if normalize_data:
            self.X = minmax_scale(self.X, axis=0) # column-wise
        (self.N, self.d) = np.shape(X)
        self.map_shape = map_shape
        self.max_iter = max_iter
        self.M = map_shape[0]*map_shape[1]  # number of nodes in the network
        self.map = np.random.random((self.map_shape[0], self.map_shape[1], self.d)) # randomly uniformly drawn between 0 and 1

        # constants for the update rule
        self.init_lr = init_lr
        self.init_radius = max(self.map_shape[0], self.map_shape[1])/2
        #self.init_radius = 1
        self.init_response = init_response
        self.time_constant = self.max_iter/np.log(self.init_radius)

    def learning_rate(self, t):
        """
        Decaying learning rate for the SOM network
        :param t: the current iteration step
        :return: the decayed learning rate
        """
        return self.init_lr*np.exp(t/self.max_iter)

    def gaussian_kernel(self, t, dist_b):
        """
        Calculates the gaussian kernel between two protoypes W_b and W_j
        :param t: current iteration step
        :param dist_b: distance between W_b and W_j on the 2D grid
        :return: the response of W_j on W_b (or vice versa)
        """
        return self.init_response*np.exp(-(dist_b**2/(2*self.radius(t)**2)))

    def radius(self, t):
        """
        Decaying radius for the gaussian kernel
        :param t: the current iteration step
        :return: the decayed radius
        """
        return self.init_radius*np.exp(-t/self.time_constant)

    def BMU(self, random_sample):
        """
        Computes the BMU for a randomly picked sample.
        :param random_sample: random input vector (X_i)
        :return: the prototype and index corresponding to the BMU.
        """
        BMU_dist = float('inf')
        BMU_index = -1
        BMU_prototype = np.array([])

        for x in range(self.map_shape[0]):
            for y in range(self.map_shape[1]):
                W_j = self.map[x, y, :]
                current_distance = euclidean_distance(random_sample, W_j)
                if (BMU_dist > current_distance):
                    BMU_dist = current_distance
                    BMU_index = np.array([x, y])
                    BMU_prototype = W_j

        return BMU_prototype, BMU_index

    def update_rule(self, t, random_sample, dist_b, W_j):
        """
        The Kohonen update rule to update a prototype.
        :param t: the current iteration step
        :param random_sample: X_i
        :param dist_b: distance between W_b and W_j on the 2D grid
        :param W_b: prototype (corresponding to the BMU)
        :param W_j: prototype
        :return:
        """

        return W_j + self.learning_rate(t) * self.gaussian_kernel(t, dist_b) * (random_sample - W_j)

    def train(self, print_progress = True):
        """
        Trains the SOM network with the data X
        """
        t0 = time()
        print("Training SOM network...")
        for t in range(self.max_iter):
            if print_progress == True:
                if t%100==0:
                    print("Iteration %d: elapsed time: %.3f"%(t, time()-t0))
            # select a random sample from X:
            random_index = np.random.randint(0, self.N)  # randomly uniformly draw an integer between 0 and N
            random_sample = self.X[random_index, :]

            # Fing the prototype and index of the BMU:
            W_b, BMU_index = self.BMU(random_sample)

            # update all the prototypes of the map by applying the kohonen update rule:
            for x in range(self.map_shape[0]):
                for y in range(self.map_shape[1]):
                    W_j = self.map[x, y, :]
                    dist_b = euclidean_distance(np.array([x,y]), BMU_index)
                    self.map[x, y, :] = self.update_rule(t, random_sample, dist_b, W_j)

        print("SOM network with %d iterations finished in %.3f seconds" % (self.max_iter, time() - t0))

    def predict(self, X):
        """
        Predicts the nearest prototypes with the corresponding indices on the 2D grid and the labels
        for a given data sample X.
        Note: the labels traverses the 2D grid from left to right starting at the first row.
        :param X: the data sample to be predicted
        :return: protottypes, indices, labels ranging from 0 to M-1
        """

        (n, d) = np.shape(X)
        indices = np.zeros((n, 2)) # x and y coordinate on 2D grid
        labels = np.zeros(n)
        prototypes = np.zeros((n, d))
        for i in range(n):
            prototypes[i, :], indices[i,:] = self.BMU(X[i,:])
            labels[i] = int(indices[i, 1] + indices[i, 0]*self.map_shape[1])

        return prototypes, indices, labels.astype(int)









