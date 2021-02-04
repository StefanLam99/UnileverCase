# from networkx.drawing.tests.test_pylab import plt
from sklearn.cluster import KMeans
from sklearn import mixture
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from two_stage_clustering import *
from DataStatistics import *
from sklearn.metrics import davies_bouldin_score



def get_davies_bouldin_score(cluster_df_np, center, clus_method):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - range of k values
        clus_method - 'kmeans' or 'gmm'
    OUTPUT:
        score - the Davies Bouldin score for the TwoStageClustering model fit to the data
    '''
    # instantiate kmeans
    TST = TwoStageClustering(cluster_df_np, n_clusters=center, clus_method= clus_method, normalize_data=False)
    # Then fit the model to your data using the fit method
    TST.train()
    model = TST.predict(cluster_df_np)

    # Calculate Davies Bouldin score

    score = davies_bouldin_score(cluster_df_np, model)

    return score

def get_DB_TSC(cluster_df_np, centers, clus_method):
    '''
    Plot the Davies-Bouldin value against the number of clusters
    :param cluster_df_np:  data
    :param centers: range of k values
    :param clus_method: select Kmeans or GMM
    :return: the Davies Boulding scores and the corresponding plot
    '''
    scores = []
    for center in centers:
        scores.append(get_davies_bouldin_score(cluster_df_np, center, clus_method))
    plt.plot(centers, scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Davies Bouldin score');
    plt.title('Davies Bouldin score vs. K ');
    plt.show()
    return scores


def get_kmeans_DB_score(cluster_df_subset, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    # instantiate kmeans
    kmeans = KMeans(n_clusters=center)
    # Then fit the model to your data using the fit method
    model = kmeans.fit_predict(cluster_df_subset)

    # Calculate Davies Bouldin score

    score = davies_bouldin_score(cluster_df_subset, model)

    return score

def get_silhouette(obs, NumberOfClusters = range(3, 10), gmmOrKmeans = 'kmeans'):
    silhouette_score_values = list()

    for i in NumberOfClusters:
        classifier = TwoStageClustering(obs, n_clusters=i, max_iter_SOM=10000, normalize_data=True, clus_method= gmmOrKmeans)
        classifier.train()
        labels = classifier.predict(obs)
        print("Number Of Clusters:")
        print(i)
        print("Silhouette score value")
        silhouette = sklearn.metrics.silhouette_score(obs, labels, metric='euclidean', sample_size=None,
                                                      random_state=None)
        print(silhouette)
        silhouette_score_values.append(silhouette)
    plt.plot(NumberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.show()

    Optimal_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(Optimal_NumberOf_Components)
    return silhouette_score_values

if __name__ == '__main__':
    #Load the data

    cluster_df = pd.read_csv("zipcodedata_version_1.csv")
    cluster_df_np = cluster_df.to_numpy()
    #cluster_df_subset = cluster_df.head(100)
    #print(cluster_df_subset)

    get_silhouette(cluster_df_np)
    get_DB_TSC(cluster_df_np, list(range(2, 10)),clus_method="kmeans")
    #model2 = KMeans(10)
    #model2.fit(cluster_df)
    #labels2 = model2.predict(cluster_df)


    #model = TwoStageClustering(cluster_df_np, max_iter_SOM=10000, clus_method = "kmeans")
    #model.train()
    #labels = model.predict(cluster_df_np)
    #statistics = get_numerical_statistics_clusters(cluster_df_np, labels)
    #means = {}
    #for i, cluster in enumerate(statistics.keys()):
     #   means[cluster] = statistics[cluster]["mean"]

    #print(means)

