# from networkx.drawing.tests.test_pylab import plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import mixture
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from two_stage_clustering import *
from DataStatistics import *
from sklearn.metrics import davies_bouldin_score
# Elbow Method for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
from Utils import *

def elbow_TSC(cluster_df, n_cluster):
    cluster_df_np = cluster_df.to_numpy()
    classifier = TwoStageClustering(cluster_df_np, max_iter_SOM=10000, normalize_data=False, clus_method="kmeans")
    classifier.train()
    W, indices, _ = classifier.model_SOM.predict(cluster_df_np)
    W = pd.DataFrame(W)
    res = list()
    for n in n_cluster:
        kmeans = GaussianMixture(n_components=n)
        # kmeans = KMeans(n_clusters=center)
        # Then fit the model to your data using the fit method
        labels = kmeans.fit_predict(W)
        # labels = kmeans.predict(cluster_df_np)
        variables = W.columns
        variable_df = []
        for variable in variables:
            # means = {}
            mean_var = []
            statistics = get_numerical_statistics_clusters(W[variable], labels)
            for i, cluster in enumerate(statistics.keys()):
                #   means[cluster] = statistics[cluster]["mean"]
                mean_var.append(statistics[cluster]["mean"])
            # print('Mean of', variable, 'is', means)
            variable_df.append(mean_var)

        df = pd.DataFrame(variable_df)
        dfT = df.T
        array = np.asarray(dfT.values.tolist())
        res.append(np.average(np.min(cdist(W, array, 'euclidean'), axis=1)))
    print(res)
    plt.plot(n_cluster, res, 'bx-')
    plt.title('elbow curve for TSC with GMM test ')
    plt.show()
    return res



def elbow_TSC_Kmeans(cluster_df,n_cluster):
    cluster_df_np = cluster_df.to_numpy()

    classifier = TwoStageClustering(cluster_df_np, max_iter_SOM=10000, normalize_data=False, clus_method="kmeans")
    classifier.train()
    W, indices, _ = classifier.model_SOM.predict(cluster_df_np)
    W = pd.DataFrame(W)
    res = list()
    for n in n_cluster:
        kmeans = KMeans(n_clusters=n)
        # Then fit the model to your data using the fit method
        labels = kmeans.fit_predict(W)
        # labels = kmeans.predict(cluster_df_np)
        variables = W.columns
        variable_df = []
        for variable in variables:
            # means = {}
            mean_var = []
            statistics = get_numerical_statistics_clusters(W[variable], labels)
            for i, cluster in enumerate(statistics.keys()):
                #   means[cluster] = statistics[cluster]["mean"]
                mean_var.append(statistics[cluster]["mean"])
            # print('Mean of', variable, 'is', means)
            variable_df.append(mean_var)
        df = pd.DataFrame(variable_df)
        dfT = df.T
        array = np.asarray(dfT.values.tolist())
        res.append(np.average(np.min(cdist(W, array, 'euclidean'), axis=1)))

    plt.plot(n_cluster, res, 'bx-')
    plt.title('elbow curve for TSC with Kmeans with correct W ')
    plt.show()
    return res

def elbowKmeans(cluster_df,n_cluster):
    #cluster_df = minmax_scale(cluster_df, axis=0)

    res = list()
    for n in n_cluster:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(cluster_df)
        res.append(np.average(np.min(cdist(cluster_df, kmeans.cluster_centers_, 'euclidean'), axis=1)))

    plt.plot(n_cluster, res,'bx-')
    plt.title('elbow curve normalized kmeans')
    plt.show()
    return res


def elbow_gmm(cluster_df,n_cluster):
    #cluster_df = minmax_scale(cluster_df, axis=0)
    #cluster_df = pd.DataFrame(cluster_df)
    res = list()
    for n in n_cluster:
        model = GaussianMixture(n_components= n)
        model.fit(cluster_df)
        labels = model.predict(cluster_df)
        variables = cluster_df.columns
        variable_df = []
        for variable in variables:
            # means = {}
            mean_var = []
            statistics = get_numerical_statistics_clusters(cluster_df[variable], labels)
            for i, cluster in enumerate(statistics.keys()):
                #   means[cluster] = statistics[cluster]["mean"]
                mean_var.append(statistics[cluster]["mean"])
            # print('Mean of', variable, 'is', means)
            variable_df.append(mean_var)

        df = pd.DataFrame(variable_df)
        dfT = df.T
        array = np.asarray(dfT.values.tolist())



        res.append(np.average(np.min(cdist(cluster_df, array, 'euclidean'), axis=1)))

    plt.plot(n_cluster, res,'bx-')
    plt.title('elbow curve normalized gmm sum' )
    plt.show()
    return res

def davies_bouldin_score_TSC(W, cluster_df_np, center):
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
    kmeans2 = GaussianMixture(n_components=center)
    kmeans = KMeans(n_clusters=center)
    # Then fit the model to your data using the fit method
    model = kmeans.fit_predict(W)
    # model = TST.predict(cluster_df_np)
    model2 = kmeans2.fit_predict(W)
    # Calculate Davies Bouldin score

    score = davies_bouldin_score(W, model)
    score2 = davies_bouldin_score(W, model2)
    return score, score2

def get_DB_TSC(cluster_df_np, centers):
    '''
    Plot the Davies-Bouldin value against the number of clusters
    :param cluster_df_np:  data
    :param centers: range of k values
    :param clus_method: select Kmeans or GMM
    :return: the Davies Boulding scores and the corresponding plot
    '''

    TST = TwoStageClustering(cluster_df_np, normalize_data=False)
    # Then fit the model to your data using the fit method
    TST.train()
    W, indices, _ = TST.model_SOM.predict(cluster_df_np)
    scores = []
    scores_gmm = []
    for center in centers:
        db_2k, db_2g = davies_bouldin_score_TSC(W, cluster_df_np, center)
        scores.append(db_2k)
        scores_gmm.append(db_2g)
    #plt.plot(centers, scores, linestyle='--', marker='o', color='b');
    #plt.xlabel('K');
    #plt.ylabel('Davies Bouldin score');
    #plt.title('Davies Bouldin score vs. K Two Stage Clustering using Kmeans');
    #plt.show()
    return scores, scores_gmm


def DB_score_Kmeans(cluster_df_subset, center):
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

def get_DB_Kmeans(cluster_df, centers= list(range(2, 11))):
    scores = []
    for center in centers:
        scores.append(DB_score_Kmeans(cluster_df, center))

    #plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    #plt.xlabel('K')
    #plt.ylabel('Davies Bouldin score')
    #plt.title('Davies Bouldin score vs. K (regular Kmeans)')
    #plt.show()
    return scores

def DB_score_GMM(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    # instantiate kmeans
    GMM = GaussianMixture(n_components=center)
    # Then fit the model to your data using the fit method
    model = GMM.fit_predict(data)

    # Calculate Davies Bouldin score

    score = davies_bouldin_score(data, model)

    return score

def get_DB_GMM(cluster_df,  centers= list(range(2, 11))):
    scores = []
    for center in centers:
        scores.append(DB_score_GMM(cluster_df, center))

    #plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    #plt.xlabel('K')
    #plt.ylabel('Davies Bouldin score')
    #plt.title('Davies Bouldin score vs. K GMM')
    #plt.show()
    return scores


def get_silhouette_2k(obs, NumberOfClusters):
    silhouette_score_values = list()
    classifier = TwoStageClustering(obs, max_iter_SOM=10000, normalize_data=False)
    classifier.train()
    W, indices, _ = classifier.model_SOM.predict(cluster_df_np)

    for i in NumberOfClusters:
        kmeans = KMeans(n_clusters=i)
        #kmeans = GaussianMixture(n_components=i)
        # kmeans = KMeans(n_clusters=center)
        labels = kmeans.fit_predict(W)
        print("Number Of Clusters:")
        print(i)
        print("Silhouette score value")
        silhouette = sklearn.metrics.silhouette_score(W, labels, metric='euclidean', sample_size=None,
                                                      random_state=None)
        print(silhouette)
        silhouette_score_values.append(silhouette)
    plt.plot(NumberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters TSC Kmeans")
    plt.show()

    Optimal_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(Optimal_NumberOf_Components)
    return silhouette_score_values

def get_silhouette_2g(obs, NumberOfClusters):
    silhouette_score_values = list()
    classifier = TwoStageClustering(obs, max_iter_SOM=10000, normalize_data=False)
    classifier.train()
    W, indices, _ = classifier.model_SOM.predict(cluster_df_np)

    for i in NumberOfClusters:
        #kmeans = KMeans(n_clusters=i)
        gmm = GaussianMixture(n_components=i)
        labels = gmm.fit_predict(W)
        print("Number Of Clusters:")
        print(i)
        print("Silhouette score value")
        silhouette = sklearn.metrics.silhouette_score(W, labels, metric='euclidean', sample_size=None,
                                                      random_state=None)
        print(silhouette)
        silhouette_score_values.append(silhouette)
    #plt.plot(NumberOfClusters, silhouette_score_values)
    #plt.title("Silhouette score values vs Numbers of Clusters TSC GMM")
    #plt.show()

    Optimal_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(Optimal_NumberOf_Components)
    return silhouette_score_values


def silhouette_kmeans(obs, NumberOfClusters=range(2, 11)):
    silhouette_score_values = list()

    for i in NumberOfClusters:
        kmeans = KMeans(n_clusters=i)
        labels = kmeans.fit_predict(obs)
        print("Number Of Clusters:")
        print(i)
        print("Silhouette score value")
        silhouette = sklearn.metrics.silhouette_score(obs, labels, metric='euclidean', sample_size=None,
                                                      random_state=None)
        print(silhouette)
        silhouette_score_values.append(silhouette)
    #plt.plot(NumberOfClusters, silhouette_score_values)
    #plt.title("Silhouette score values vs Numbers of Clusters Kmeans")
    #plt.show()

    Optimal_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(Optimal_NumberOf_Components)
    return silhouette_score_values


def silhouette_gmm(obs, NumberOfClusters=range(2, 11)):
    silhouette_score_values = list()

    for i in NumberOfClusters:
        kmeans = GaussianMixture(n_components=i)
        labels = kmeans.fit_predict(obs)
        print("Number Of Clusters:")
        print(i)
        print("Silhouette score value")
        silhouette = sklearn.metrics.silhouette_score(obs, labels, metric='euclidean', sample_size=None,
                                                      random_state=None)
        print(silhouette)
        silhouette_score_values.append(silhouette)
    #plt.plot(NumberOfClusters, silhouette_score_values)
    #plt.title("Silhouette score values vs Numbers of Clusters GMM")
    #plt.show()

    Optimal_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(Optimal_NumberOf_Components)
    return silhouette_score_values

def elbow_tsc_kmeans(obs):
    classifier = TwoStageClustering(obs, max_iter_SOM=10000, normalize_data=False, clus_method="kmeans")
    classifier.train()
    W, indices, _ = classifier.model_SOM.predict(cluster_df_np)
    distortions = []
    res = []
    K = range(2, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(W)
        distortions.append(kmeanModel.inertia_)
        res.append(np.sum(np.min(cdist(W, kmeanModel.cluster_centers_, 'euclidean'), axis=1)))

    # plt.figure(figsize=(16, 8))
    print(distortions)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k with TSC')
    plt.show()

    plt.plot(K, res, 'bx-')
    plt.show()
    return distortions

def elbow_kmeans(df):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
    print(distortions)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k with regular Kmeans DIS ONE')
    plt.show()
    return distortions

if __name__ == '__main__':
    #Load the data

    cluster_df = pd.read_csv("Data/zipcodedata_KNN_normalized_version_6.csv")
    cluster_df_np = cluster_df.to_numpy()
    #cluster_df = minmax_scale(cluster_df, axis=0)

   # DAVIES BOULDIN SCORE
    rangeK = list(range(2, 9))
    scores1 = get_DB_Kmeans(cluster_df, centers= rangeK)
    scores2 = get_DB_GMM(cluster_df, centers= rangeK)
    k2, g2 = get_DB_TSC(cluster_df_np, centers = rangeK)


    # SILHOUETTE FOR TSC

    res1 = get_silhouette_2k(cluster_df_np, NumberOfClusters=rangeK)
    res2 = get_silhouette_2g(cluster_df_np, NumberOfClusters=rangeK)
    res3 = silhouette_kmeans(cluster_df, NumberOfClusters=rangeK)
    res4 = silhouette_gmm(cluster_df, NumberOfClusters=rangeK)
    #model2 = KMeans(10)
    #model2.fit(cluster_df)
    #labels2 = model2.predict(cluster_df)

    #distortions = elbow_tsc_kmeans(cluster_df_np)
    #distortions = elbow_kmeans(cluster_df)
    K = range(2, 9)
    plt.plot(K, res2, 'cx-', linestyle='--', label="TSC - GMM")
    plt.plot(K, res3, 'rx-', linestyle='--', label="Kmeans")
    plt.plot(K, res1, 'gx-', linestyle='--', label="TSC - Kmeans")
    plt.plot(K, res4, 'mx-', linestyle='--', label="GMM")
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for the clustering methods ')
    plt.legend()
    plt.show()

    plt.plot(K, scores1, 'cx-', label="Kmeans")
    plt.plot(K, scores2, 'rx-', label="GMM")
    plt.plot(K, g2, 'gx-', label="TSC - GMM")
    plt.plot(K, k2, 'mx-', label="TSC - Kmeans")
    plt.xlabel('Number of clusters')
    plt.ylabel('DB score')
    plt.title('Davies-Bouldin score for the clustering methods ')
    plt.legend()
    plt.show()

    silhouettes_df = pd.DataFrame({'K': K, 'Kmeans': res3, 'GMM': res4, 'TSC - Kmeans': res1, 'TSC - GMM': res2}).round(3)
    silhouettes_df = silhouettes_df.set_index('K')
    make_latex_table(silhouettes_df, scale=0.8)

    DB_df = pd.DataFrame({'K': K, 'Kmeans': scores1, 'GMM': scores2, 'TSC - Kmeans': k2, 'TSC - GMM': g2}).round(3)
    DB_df = DB_df.set_index('K')
    make_latex_table(DB_df, scale=0.8)
    elbow_tsc_gmm = elbow_TSC(cluster_df, K)
    elbow_gmm = elbow_gmm(cluster_df, K)
    elbow_km = elbowKmeans(cluster_df, K)
    elbow_tsc_km = elbow_TSC_Kmeans(cluster_df, K)

    plt.plot(K, elbow_gmm, 'cx-', label="GMM")
    plt.plot(K, elbow_km, 'rx-',label="Kmeans")
    plt.plot(K, elbow_tsc_gmm, 'gx-', label="TSC - GMM")
    plt.plot(K, elbow_tsc_km, 'mx-', label="TSC - Kmeans")
    plt.xlabel('Number of clusters')
    plt.ylabel('WSS')
    plt.title('Elbow method for the clustering methods ')
    plt.legend()
    plt.show()
