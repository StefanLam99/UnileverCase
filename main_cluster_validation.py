import numpy as np
from two_stage_clustering import TwoStageClustering
from ClusteringValidationMetrics import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from SOM import SOM
from Utils import *
from sklearn.metrics import davies_bouldin_score, silhouette_score
from time import time


def main_validation(version):
    '''
    Main to obtain the cluster validation measures (silhouette score and DB-index) for the implemented clustering methods
    '''

    # data
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")
    data_normalised, _, _ = normalise(data)
    X = data_normalised.iloc[:,1:].values  # exclude pc4 variable

    # parameters
    k_range = np.r_[2:21]
    map_shape = (8, 8)

    # measures
    DB_measures = np.zeros((len(k_range), 4))  # rows the k, colums the models (order: TSC-kmeanss, TSC-GMM, kmeans, GMM)
    silhouette_measures = np.zeros((len(k_range), 4))  # rows the k, colums the models

    # models
    model_SOM = SOM(X=X, map_shape=map_shape)
    model_SOM.train(print_progress=True)
    W = model_SOM.map  # use this to train the kmeans and GMM for the TSC
    for i, k in enumerate(k_range):
        print("CURRENT k = %d" % k)
        models = [TwoStageClustering(X=X, W=W, n_clusters=k, map_shape=map_shape),
                  TwoStageClustering(X=X, W=W, n_clusters=k, clus_method="gmm", map_shape=map_shape),
                  KMeans(n_clusters=k, random_state=0, algorithm="full", max_iter=5000, n_init=10),
                  GaussianMixture(n_components=k, max_iter=5000, n_init=10, init_params="random")]

        for j, model in enumerate(models):
            if j < 2:  # first two models are two-stage models
                model.train(print_progress=False)
            elif j == 2:
                print("Training k-means....")
                t0 = time()
                model.fit(X)
                print("The k-means algorithm took %.3f seconds" % (time()-t0 ))
            elif j == 3:
                print("Training GMM....")
                t0 = time()
                model.fit(X)
                print("The GMM algorithm took %.3f seconds" % (time()-t0 ))

            labels = model.predict(X)
            DB_measures[i,j] = davies_bouldin_score(X, labels)
            silhouette_measures[i, j] = silhouette_score(X, labels)
        print("")

    np.savetxt("Results/DB_measures.txt", DB_measures, delimiter=',')
    np.savetxt("Results/silhouette_measures.txt", silhouette_measures, delimiter=',')


def main_visul():
    '''
    Main to visualize the validation measures
    '''
    DB_measures = np.genfromtxt("Results/DB_measures.txt", delimiter=',')
    silhouette_measures = np.genfromtxt("Results/silhouette_measures.txt", delimiter=',')
    k_range = np.r_[2:21]
    k_indices = np.r_[0:19]
    #  plot Davies Bouldin measures
    plt.plot(k_range, DB_measures[:, 0], 'cx-', label="TSC - Kmeans", marker = 'o', markevery = [i for i in  k_indices[1:]])
    plt.plot(k_range, DB_measures[:, 0], 'cx-', marker = '*', markevery = [k_indices[0]], markersize=14)

    plt.plot(k_range, DB_measures[:, 1], 'rx-', label="TSC - GMM", marker = 'o', markevery = [i for i in  k_indices[1:]])
    plt.plot(k_range, DB_measures[:, 1], 'rx-', marker='*', markevery=[k_indices[0]], markersize=14)

    plt.plot(k_range, DB_measures[:, 2], 'gx-', label="Kmeans", marker = 'o', markevery = [i for i in  k_indices[1:]])
    plt.plot(k_range, DB_measures[:, 2], 'gx-', marker='*', markevery=[k_indices[0]], markersize=14)

    plt.plot(k_range, DB_measures[:, 3], 'mx-', label="GMM", marker = 'o', markevery = [i for i in  k_indices[1:]])
    plt.plot(k_range, DB_measures[:, 3], 'mx-', marker='*', markevery=[k_indices[0]], markersize=14)

    plt.xlabel('Number of clusters', fontsize= 14)
    plt.ylabel('DB Index', fontsize= 14)
    plt.xticks(k_range)
    #plt.title('Davies-Bouldin score for the clustering methods version ' + str(version))
    #plt.legend(fontsize=12, fancybox=True, edgecolor = "black", frameon=True)
    plt.savefig("Figures/Plots/DB_validation_plot.png")
    plt.show()

    #  plot Silhouette measures
    plt.plot(k_range, silhouette_measures[:, 0], 'cx-', label="TS k-means", marker = 'o', markevery=[i for i in  k_indices[1:]])
    plt.plot(k_range, silhouette_measures[:, 0], 'cx-', marker = '*', markevery = [k_indices[0]], markersize=14)

    plt.plot(k_range, silhouette_measures[:, 1], 'rx-', label="TS GMM", marker = 'o', markevery=[i for i in  k_indices[1:]])
    plt.plot(k_range, silhouette_measures[:, 1], 'rx-', marker = '*', markevery = [k_indices[0]], markersize=14)

    plt.plot(k_range, silhouette_measures[:, 2], 'gx-', label="k-means", marker = 'o', markevery=[i for i in  k_indices[1:]])
    plt.plot(k_range, silhouette_measures[:, 2], 'gx-', marker='*', markevery = [k_indices[0]], markersize=14)

    plt.plot(k_range, silhouette_measures[:, 3], 'mx-', label="GMM", marker = 'o', markevery=[i for i in  k_indices[1:]])
    plt.plot(k_range, silhouette_measures[:, 3], 'mx-', marker='*', markevery = [k_indices[0]], markersize=14)

    plt.xlabel('Number of clusters', fontsize= 14)
    plt.ylabel('Average Silhouette Coefficient', fontsize=14)
    plt.xticks(k_range)
    #plt.title('Silhouette score for the clustering methods version ' + str(version))
    plt.legend(fontsize=12, fancybox=True, edgecolor = "black", frameon =True)
    plt.savefig("Figures/Plots/Silhouette_validation_plot.png")
    plt.show()


if __name__ == '__main__':
    version = 10
    main_validation(version)
    main_visul()
