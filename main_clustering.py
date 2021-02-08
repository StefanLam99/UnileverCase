import numpy as np
from two_stage_clustering import TwoStageClustering
from DataStatistics import tSNE_visualisation, PCA_visualisation
from ClusteringValidationMetrics import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale, normalize
from sklearn import preprocessing
from SOM import SOM
from time import time
from DataStatistics import *
from DataSets import Neighborhood_Descriptives
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break


def main():
    # data
    data = pd.read_csv("Data/zipcodedata_version_1.csv")
    data= pd.read_csv("Data/zipcodedata_MICE_nonzeros.csv")
    print(data)
    X = np.array(data.iloc[:,1:])
    #X = minmax_scale(X, axis=0)
    X = normalize(X, axis=0)
    neighbor_data = Neighborhood_Descriptives()
    np.random.seed(0)
    random_clusters = np.random.randint(0,3,np.shape(neighbor_data)[0])
    a = get_categorical_counts_clusters_df(neighbor_data, random_clusters, var_names=["DEGURBA", "higher_education"])
    print(random_clusters)
    print(a)
    # models
    model_SOM = SOM(X=X, map_shape=[2,2], max_iter=10000)
    model_twoStage = TwoStageClustering(X=X, map_shape=[5,5], n_clusters=4)
    model_twoStage.train(print_progress=False)
    #model_SOM.train(print_progress=False)
    #prototypes, indices, labels = model_SOM.predict(X)
    labels = model_twoStage.predict(X)
    print(labels)
    print(type(labels))
    print(set(labels))

    result = get_numerical_statistics_clusters_df(data, labels, var_names=["MAN", 'INW_014','AANTAL_HH' ])
    result.to_csv('haai.csv')
    print(result)


def main_visualisation():
    # data
    data = pd.read_csv("Data/zipcodedata_KNN_normalized.csv")
    X = data.iloc[:,1:].values  # exclude pc4 variable

    # model two stage k means
    opt_k = 4  #  optimal number of clusters
    model = TwoStageClustering(X=X, n_clusters=opt_k)
    model.train()
    labels = model.predict(X)

    tSNE_visualisation(X, labels, title="t-SNE with twostage-kmeans clustering")
    PCA_visualisation(X, labels, title="PCA with twostage-kmeans clustering")

    # model two stage GMM
    opt_k = 6  #  optimal number of clusters
    model = TwoStageClustering(X=X, n_clusters=opt_k, clus_method="gmm")
    model.train()
    labels = model.predict(X)

    tSNE_visualisation(X, labels, title="t-SNE with twostage-GMM clustering")
    PCA_visualisation(X, labels, title="PCA with twostage-GMM clustering")

    # model k means
    opt_k = 5  #  optimal number of clusters
    model = KMeans(n_clusters=opt_k, random_state=0, algorithm="full", max_iter=5000, n_init=10)
    model.fit(X)
    labels = model.predict(X).astype(int)
    tSNE_visualisation(X, labels, title="t-SNE with kmeans clustering")
    PCA_visualisation(X, labels, title="PCA with kmeans clustering")

    # model GMM
    opt_k = 5  #  optimal number of clusters
    model = GaussianMixture(n_components=opt_k, max_iter=5000, n_init=10, init_params="random")
    model.fit(X)
    labels = model.predict(X).astype(int)
    tSNE_visualisation(X, labels, title="t-SNE with GMM clustering")
    PCA_visualisation(X, labels, title="PCA with GMM clustering")

    ''' 
    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.scatter(x=X_embedded[:,0],y=X_embedded[:,1], cmap='Paired', c=labels)
    plt.title("t-SNE with twostage-kmeans clustering")
    plt.show()
'''


def main3():
    data= pd.read_csv("Data/zipcodedata_version_2_nanIncluded.csv")
    print(data)


if __name__ == '__main__':
    #main3()
    main_visualisation()


