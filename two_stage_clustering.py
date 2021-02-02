import numpy as np
from Utils import hit_rate, infer_cluster_labels, infer_data_labels, make_dir
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize, minmax_scale
from SOM import SOM
from time import time
import pickle as pkl
from sklearn.datasets import load_digits
#from keras.datasets import mnist
import SimpSOM as sps
import sys
#np.set_printoptions(threshold=sys.maxsize)


class TwoStageClustering:
    """
    Class to make a two-stage clustering model, where the first stage is a SOM network and the second stage
    is a clustering method such as k-means or GMM.
    """

    def __init__(self, X, map_shape=(5,5), n_clusters=10, init_lr=0.01, init_response=1, max_iter_SOM=20000, max_iter_clus=1000, clus_method="kmeans", normalize_data=False, seed=0):

        # data and SOM map shape
        self.X = X
        if normalize_data:
            self.X = minmax_scale(self.X, axis=0)  # column-wise
        (self.N, self.d) = np.shape(X)
        self.map_shape = map_shape
        self.M = map_shape[0] * map_shape[1]  # number of nodes in the network

        # hyperparameters
        self.max_iter_SOM = max_iter_SOM
        self.max_iter_clus = max_iter_clus
        self.seed =seed
        self.n_clusters = n_clusters
        self.init_lr = init_lr
        self.init_response = init_response

        # first stage model
        self.model_SOM = SOM(X=self.X, map_shape=self.map_shape, init_lr=self.init_lr, init_response=self.init_response,
                             max_iter=self.max_iter_SOM, seed=self.seed)

        #  second stage model
        self.clus_method = clus_method
        if self.clus_method == "kmeans":
            self.model_clus = KMeans(n_clusters=self.n_clusters, random_state=self.seed, algorithm="full", max_iter=self.max_iter_clus)
        else:
            self.model_clus = GaussianMixture(n_components=self.n_clusters, max_iter=self.max_iter_clus)

    def train(self, print_progress=True):
        """
        First trains the SOM network, then the second stage model with the prototypes from the SOM network.
        """

        # training first stage SOM network
        print("Start training the two stage clustering procedure...")
        t0 = time()  # starting time training SOM
        self.model_SOM.train(print_progress=print_progress)
        W = self.model_SOM.map  # 3D array containing the M prototypes

        # fitting second stage clustering method
        t1 = time()  # starting time second stage clustering method
        print("Training %s clustering method..." % self.clus_method)
        self.model_clus.fit(W.reshape((self.M, self.d)))  # reshape to a (M, d) matrix
        print("%s clustering method with %d iterations finished in %.3f seconds" %(self.clus_method, self.max_iter_clus, time()-t1))
        print('\n')
        print("The two stage clustering procedure with %s took %.3f" % (self.clus_method, time()-t0))

    def predict(self, X):
        """
        Predicts the labels of X with the two stage clustering procedure. First, get the corresponding prototype of each sample
        of X, then predict the label of the prototype with the clustering method.
        :param X: the data sample to be predicted
        :return: the predicted labels
        """
        W, indices, _ = self.model_SOM.predict(X)
        labels = self.model_clus.predict(W)
        return labels.astype(int)

    def save(self, file_name=None):
        """
        Method to save the model as a pickle file
        """
        if file_name == None:
            print("No file name is given!!!!")
            return
        dir_name = "Models/TwoStageClustering/"
        make_dir(dir_name)
        filehandler = open(dir_name + file_name+".pkl", "wb")
        pkl.dump(self, filehandler)
        filehandler.close()


def load(file_name=None):
    """
    Method to load the two stage clustering model from a pickle file
    """
    if file_name == None:
        print("No file name is given!!!!")
        return
    dir_name = "Models/TwoStageClustering/"
    filehandler = open(dir_name + file_name + ".pkl", "rb")
    model = pkl.load(filehandler)
    filehandler.close()
    return model

