import numpy as np
from two_stage_clustering import TwoStageClustering
from ClusteringValidationMetrics import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats
from sklearn.preprocessing import minmax_scale
from Utils import *
from SOM import SOM
from time import time
from DataStatistics import *
from DataSets import Neighborhood_Descriptives
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break


#ef main_results():
#   # data
#   data = pd.read_csv("Data/zipcodedata_KNN.csv")
#   X = data.iloc[:,1:].values  # exclude pc4 variable
#   #X = minmax_scale(X, axis=0)
#   X = normalize(X, axis=0)
#   neighbor_data = Neighborhood_Descriptives()
#   np.random.seed(0)
#   random_clusters = np.random.randint(0,3,np.shape(neighbor_data)[0])
#   a = get_categorical_counts_clusters_df(neighbor_data, random_clusters, var_names=["DEGURBA", "higher_education"])
#   print(random_clusters)
#   print(a)
#   # models
#   model_SOM = SOM(X=X, map_shape=[2,2], max_iter=10000)
#   model_twoStage = TwoStageClustering(X=X, map_shape=[5,5], n_clusters=4)
#   model_twoStage.train(print_progress=False)
#   #model_SOM.train(print_progress=False)
#   #prototypes, indices, labels = model_SOM.predict(X)
#   labels = model_twoStage.predict(X)
#   print(labels)
#   print(type(labels))
#   print(set(labels))

#   result = get_numerical_statistics_clusters_df(data, labels, var_names=["MAN", 'INW_014','AANTAL_HH' ])
#   result.to_csv('haai.csv')
#   print(result)

def main_statistics():
    '''
    Test some statistics of our "best model": SOM + kmeans
    '''
    # data
    k = 4
    data = pd.read_csv("Data/zipcodedata_KNN.csv")
    print(data)
    var_names = ["INWONER", "P_INW_014", "P_INW_1524", "P_INW_2544", "P_INW_4564", "P_INW_65PL", "P_NL_ACHTG", "P_WE_MIG_A", "P_NW_MIG_A",
                 "GEM_HH_GR", "UITKMINAOW", "OAD", "P_LINK_HH", "P_HINK_HH", "AV5_CAFE", "AV5_CAFTAR", "AV5_HOTEL", "AV5_RESTAU", "log(median_inc)"]
    stat_names = ["mean", "std"]
    data["log(median_inc)"] = np.log(data["median_inc"])
    data = data.drop("median_inc", 1)
    total_ages = data["INW_014"] + data["INW_1524"] +data["INW_2544"] +data["INW_4564"] +data["INW_65PL"]
    data["P_INW_014"] = data["INW_014"]/total_ages
    data["P_INW_1524"] = data["INW_1524"]/total_ages
    data["P_INW_2544"] = data["INW_2544"]/total_ages
    data["P_INW_4564"] = data["INW_4564"]/total_ages
    data["P_INW_65PL"] = data["INW_65PL"]/total_ages

    data = data.drop("INW_014", 1)
    data = data.drop("INW_1524", 1)
    data = data.drop("INW_2544", 1)
    data = data.drop("INW_4564", 1)
    data = data.drop("INW_65PL", 1)
    print(data)
    data_normalised = pd.read_csv("Data/zipcodedata_KNN_normalized.csv")

    data_normalised, _, _ = normalise(data)
    X = data_normalised.iloc[:,1:].values  # exclude pc4 variable
    print(data_normalised)
    print(data_normalised[["P_INW_65PL", "P_INW_4564"]])
    model = TwoStageClustering(X=X, n_clusters=k)
    model.train(print_progress=False)
    labels = model.predict(X)
    print(data["P_INW_014"][labels == 0])
    print(len(data["P_INW_014"][labels == 0]))
    print(np.sum(labels==0))
    print(labels)
    columns = data.columns
    for col in columns:
        KW_test = stats.kruskal(data[col][labels==0], data[col][labels==1], data[col][labels==2], data[col][labels==3])
        ANOVA_test = stats.f_oneway(data[col][labels==0], data[col][labels==1], data[col][labels==2], data[col][labels==3])
        print("The variable %s has a KW statistic = %.3f with p-value = %.3f" %(col, KW_test[0], KW_test[1]))
        print("The variable %s has a ANOVE statistic = %.3f with p-value = %.3f" %(col, ANOVA_test[0], ANOVA_test[1]))
    df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
    print(df)


def main_results():


    # data
    data = pd.read_csv("Data/zipcodedata_KNN_version_4.csv")
    data_normalised, _, _ = normalise(data)
    X = data_normalised.iloc[:,1:].values  # exclude pc4 variable

    # parameters
    opt_k = [2, 2, 2, 2]   # the optimal k in the order: two-stage-kmeans, two-stage-GMM, kmeans, GMM
    var_names = data.columns[1:]  # exclude pc4
    stat_names = ["mean", "std"]
    titles = ["SOM + k-means", "SOM + GMM", "k-means", "GMM"]
    s = 2  # size of points in visualisations

    # embed the data into two dimensions with t-SNE and PCA
    file_path_tSNE = "Figures/Plots/tSNE_"
    file_path_PCA = "Figures/Plots/PCA_"
    X_tSNE = TSNE(n_components=2, random_state=0).fit_transform(X)
    X_PCA = PCA(n_components=2, random_state=0).fit_transform(X)

    # models
    models = [TwoStageClustering(X=X, n_clusters=opt_k[0]),
              TwoStageClustering(X=X, n_clusters=opt_k[1], clus_method="gmm"),
              KMeans(n_clusters=opt_k[2], random_state=0, algorithm="full", max_iter=5000, n_init=10),
              GaussianMixture(n_components=opt_k[3], max_iter=5000, n_init=10, init_params="random")]

    for i, model in enumerate(models):
        if i < 2:  # first two models are two-stage models
            model.train(print_progress=False)
        else:
            model.fit(X)

        #  show the statistics of the clusters
        labels = model.predict(X)
        df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
        print(df)
        make_latex_table_MultiIndex(df)
        print('')

        # plot the clusters with pca and t-SNE
        plot_clusters(X_tSNE, labels, title=titles[i], save_path=file_path_tSNE + titles[i], s=s)
        plot_clusters(X_PCA, labels, title=titles[i], save_path=file_path_PCA + titles[i], s=s)

        #  save labels:
        pc4_labels = np.concatenate(
            (data['pc4'].values.reshape(np.shape(data)[0], 1), labels.reshape(np.shape(data)[0], 1)), axis=1)
        pd.DataFrame(pc4_labels, columns=['pc4', 'labels']).astype(int).to_csv('Results/pc4_best_labels_'+titles[i]+'.csv',
                                                                               index=False)

    ''' 
    # model two stage k means
    model = TwoStageClustering(X=X, n_clusters=opt_k[0])
    model.train(print_progress=False)
    labels = model.predict(X)
    df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
    plot_clusters(X_tSNE, labels, title=titles[0], save_path=file_path_tSNE + titles[0], s=s)
    plot_clusters(X_PCA, labels, title=titles[0], save_path=file_path_PCA + titles[0], s=s)
    print(df)
    make_latex_table_MultiIndex(df)
    print('')




    # model two stage GMM
    model = TwoStageClustering(X=X, n_clusters=opt_k[1], clus_method="gmm")
    model.train(print_progress=False)
    labels = model.predict(X)
    df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
    plot_clusters(X_tSNE, labels, title=titles[1], save_path=file_path_tSNE + titles[1], s=s)
    plot_clusters(X_PCA, labels, title=titles[1], save_path=file_path_PCA + titles[1], s=s)
    print(df)
    make_latex_table_MultiIndex(df)
    print('')

    #  best labels:
    pc4_labels = np.concatenate((data['pc4'].values.reshape(np.shape(data)[0],1), labels.reshape(np.shape(data)[0],1)), axis=1)
    pd.DataFrame(pc4_labels, columns=['pc4', 'labels']).astype(int).to_csv('Results/pc4_best_labels.csv', index=False)


    # model k means
    print("Start training k-means model...")
    t0 = time()
    model = KMeans(n_clusters=opt_k[2], random_state=0, algorithm="full", max_iter=5000, n_init=10)
    model.fit(X)
    print("k-means with %d random initialisations took: %.2f seconds " % (10, time() - t0))
    labels = model.predict(X).astype(int)
    df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
    plot_clusters(X_tSNE, labels, title=titles[2], save_path=file_path_tSNE + titles[2], s=s)
    plot_clusters(X_PCA, labels, title=titles[2], save_path=file_path_PCA + titles[2], s=s)
    print(df)
    make_latex_table_MultiIndex(df)
    print('')

    # model GMM
    print("Start training GMM model...")
    t0 = time()
    model = GaussianMixture(n_components=opt_k[3], max_iter=5000, n_init=10, init_params="random")
    model.fit(X)
    print("GMM with %d random initialisations took: %.2f seconds " % (10, time() - t0))
    labels = model.predict(X).astype(int)
    df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
    plot_clusters(X_tSNE, labels, title=titles[3], save_path=file_path_tSNE + titles[3], s=s)
    plot_clusters(X_PCA, labels, title=titles[3], save_path=file_path_PCA + titles[3], s=s)
    print(df)
    make_latex_table_MultiIndex(df)
'''





def main_results_best_model():
    # data
    version = 4  # version of data
    best_method = "SOM + k-means"
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")
    labels = pd.read_csv('Results/pc4_best_labels_' + str(best_method)+ '.csv')['labels']


    # variables per table
    demographic_names = ['INWONER', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
       'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'UITKMINAOW',
       'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
    demographic_general_names = ['INWONER', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
    demographic_income_names = [ 'log_median_inc',
       'P_LINK_HH', 'P_HINK_HH', 'UITKMINAOW']
    demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
       'P_INW_4564', 'P_INW_65PL']
    neighborhood_names = ['AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']

    tables = [demographic_names, demographic_general_names, demographic_income_names, demographic_age_names, neighborhood_names]
    captions = ["Demographics of the clusters", "General demographics of the clusters", "Age demographics of the clusters",
                "Income demographics of the clusters", "Neighborhood descriptives of the clusters"]

    groups = ["General Demographics", "Age Demographics", "Income Demographics", "Neighborhood Descriptives"]
    group_tables = [demographic_general_names, demographic_age_names, demographic_income_names, neighborhood_names]
    # create dictionary with the variables:
    dict_variables = {}
    for group, table in zip(groups, group_tables):
        dict_variables[group] = table

    # parameters for tables
    scale = 0.5
    stat_names = ["mean", "median", "std"]


    for table, caption in zip(tables, captions):
        df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=table,
                                                              stat_names=stat_names)
        print(df)
        make_latex_table_MultiIndex(df, caption=caption, scale=scale)
        print('')


    df = get_numerical_statistics_clusters_df_inversed(data, clusters=labels, dict_var_names=dict_variables,
                                                              stat_names=stat_names)
    print(df)
    make_latex_table_MultiIndex_inversed(df, dict_var_names=dict_variables, caption= "Statistics of the overall zipcodes and the clusters obtained by " + str(best_method), scale=scale)
    print('')

'''

    df_demographic = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=demographic_names, stat_names=stat_names)
    print(df_demographic)
    make_latex_table_MultiIndex(df_demographic, caption="Demographics of the clusters", scale=scale)
    print('')

    df_demographic_general = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=demographic_general_names, stat_names=stat_names)
    print(df_demographic_general)
    make_latex_table_MultiIndex(df_demographic_general, caption="General demographics of the clusters", scale=scale)
    print('')

    df_demographic_age = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=demographic_age_names, stat_names=stat_names)
    print(df_demographic_age)
    make_latex_table_MultiIndex(df_demographic_age, caption="Age demographics of the clusters", scale=scale)
    print('')

    df_demographic_income = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=demographic_income_names, stat_names=stat_names)
    print(df_demographic_income)
    make_latex_table_MultiIndex(df_demographic_income, caption="Income demographics of the clusters", scale=scale)
    print('')

    df_neighborhood = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=neighborhood_names, stat_names=stat_names)
    print(df_neighborhood)
    make_latex_table_MultiIndex(df_neighborhood, caption="Neighborhood descriptives of the clusters", scale=scale)
'''


if __name__ == '__main__':
    main_results()
    main_results_best_model()
    #main_statistics()

