import numpy as np
from two_stage_clustering import TwoStageClustering
from ClusteringValidationMetrics import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from SOM import SOM
from Utils import *
from sklearn.metrics import davies_bouldin_score, silhouette_score


def main_validation(version):

    # data
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")
    data_normalised, _, _ = normalise(data)
    X = data_normalised.iloc[:,1:].values  # exclude pc4 variable

    # parameters
    titles = ["SOM + k-means", "SOM + GMM", "k-means", "GMM"]
    k_range = np.r_[2:11]

    # models
    DB_measures = np.zeros(((len(k_range), 4)))  # rows the k, colums the models (order: TSC-kmeanss, TSC-GMM, kmeans, GMM)
    silhouette_measures = np.zeros(((len(k_range), 4)))  # rows the k, colums the models
    model_SOM = SOM(X=X)
    model_SOM.train(print_progress=True)
    W = model_SOM.map  # use this to train the kmeans and GMM for the TSC
    for i, k in enumerate(k_range):
        print(k)
        models = [TwoStageClustering(X=X, W=W, n_clusters=k),
                  TwoStageClustering(X=X, W=W, n_clusters=k, clus_method="gmm"),
                  KMeans(n_clusters=k, random_state=0, algorithm="full", max_iter=5000, n_init=10),
                  GaussianMixture(n_components=k, max_iter=5000, n_init=10, init_params="random")]



        for j, model in enumerate(models):
            if j < 2:  # first two models are two-stage models
                model.train(print_progress=False)
            else:
                model.fit(X)

            labels = model.predict(X)
            print(set(labels))
            DB_measures[i,j] = davies_bouldin_score(X, labels)
            silhouette_measures[i, j] = silhouette_score(X, labels)

    #  plot Davies Bouldin measures
    plt.plot(k_range, DB_measures[:, 0], 'cx-', label="TSC - Kmeans")
    plt.plot(k_range, DB_measures[:, 1], 'rx-', label="TSC - GMM")
    plt.plot(k_range, DB_measures[:, 2], 'gx-', label="Kmeans")
    plt.plot(k_range, DB_measures[:, 3], 'mx-', label="GMM")
    plt.xlabel('Number of clusters')
    plt.ylabel('DB score')
    plt.title('Davies-Bouldin score for the clustering methods ')
    plt.legend()
    plt.show()

    #  plot Silhouette measures
    plt.plot(k_range, silhouette_measures[:, 0], 'cx-', label="TSC - Kmeans")
    plt.plot(k_range, silhouette_measures[:, 1], 'rx-', label="TSC - GMM")
    plt.plot(k_range, silhouette_measures[:, 2], 'gx-', label="Kmeans")
    plt.plot(k_range, silhouette_measures[:, 3], 'mx-', label="GMM")
    plt.xlabel('Number of clusters')
    plt.ylabel('DB score')
    plt.title('Silhouette score for the clustering methods ')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    version = 7
    main_validation(version)

