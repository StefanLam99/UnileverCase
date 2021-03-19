import numpy as np
from two_stage_clustering import TwoStageClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from Utils import *
from DataStatistics import *
from DataSets import UFS_Universe_NLnew
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break


def main_statistics(version):
    '''
    Test some statistics of our "best model": SOM + kmeans
    '''
    # data
    best_method = "SOM + k-means"
    k = 4
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")

    var_names = ["INWONER_HH", "P_INW_014", "P_INW_1524", "P_INW_2544", "P_INW_4564", "P_INW_65PL", "P_NL_ACHTG", "P_WE_MIG_A", "P_NW_MIG_A",
                 "GEM_HH_GR", "UITKMINAOW_HH", "OAD", 'AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD',  "P_LINK_HH", "P_HINK_HH", "log_median_inc"]
    stat_names = ["mean", "std"]
    labels = pd.read_csv('Results/pc4_best_labels_version_' + str(version)+ "_" + str(best_method)+ '.csv')['labels']


    columns = data.columns
    for col in columns:
        KW_test = stats.kruskal(data[col][labels==0], data[col][labels==1])
        ANOVA_test = stats.f_oneway(data[col][labels==0], data[col][labels==1])
        print("The variable %s has a KW statistic = %.3f with p-value = %.3f" %(col, KW_test[0], KW_test[1]))
        print("The variable %s has a ANOVE statistic = %.3f with p-value = %.3f" %(col, ANOVA_test[0], ANOVA_test[1]))
    df = get_numerical_statistics_clusters_df(data, clusters=labels, var_names=var_names, stat_names=stat_names)
    print(df)


def main_train(version):
    """
    Main to train the models and obtain the corresponding labels, also plots the t-SNE and PCA visualisations
    :param version:
    :return:
    """
    # data
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")
    data_normalised, _, _ = normalise(data)
    X = data_normalised.iloc[:,1:].values  # exclude pc4 variable

    # parameters
    opt_k = [2, 2, 2, 2]   # the optimal k in the order: two-stage-kmeans, two-stage-GMM, kmeans, GMM
    var_names = data.columns[1:]  # exclude pc4
    stat_names = ["mean", "std"]
    titles = ["SOM + k-means", "SOM + GMM", "k-means", "GMM"]
    titles_plots = ["TS k-means", "TS GMM", "k-means", "GMM"]
    s = 4  # size of points in visualisations

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
        plot_clusters(X_tSNE, labels+1, title=titles_plots[i], save_path=file_path_tSNE + titles[i], s=s)
        plot_clusters(X_PCA, labels+1, title=titles_plots[i], save_path=file_path_PCA + titles[i], s=s, loc="upper right")

        #  save labels:
        pc4 = data['pc4'].values.reshape(np.shape(data)[0], 1)
        labels = labels.reshape(np.shape(data)[0], 1)
        ''' 
        labels[labels==0] = -1
        labels[labels==1] = 0
        labels[labels==-1] = 1
        '''
        pc4_labels = np.concatenate(
            (pc4, labels), axis=1)
        pd.DataFrame(pc4_labels, columns=['pc4', 'labels']).astype(int).to_csv('Results/pc4_best_labels_version_' +str(version) + '_'+titles[i]+'.csv',
                                                                               index=False)


def main_results_best_model(version):
    '''
    main to obtain the latex tables for the best model containing the statistics of each cluster
    '''
    # data
    best_method = "SOM + k-means"  # best method
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")
    labels = pd.read_csv('Results/pc4_best_labels_version_' + str(version)+ "_" + str(best_method)+ '.csv')['labels'] +1


    # variables per table
    if version == 4:
        demographic_names = ['INWONER', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'UITKMINAOW',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['INWONER', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'UITKMINAOW']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ['AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']
    elif version == 5:
        demographic_names = ['INWONER_HH', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'UITKMINAOW_HH',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['INWONER_HH', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'UITKMINAOW_HH']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ['AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']
    elif version == 6:
        demographic_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'UITKMINAOW_HH',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'UITKMINAOW_HH']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ['AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']
    elif version == 7:
        demographic_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'P_UITKMINAOW',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'P_UITKMINAOW']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ['AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']
    elif version == 8:
        demographic_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'P_UITKMINAOW',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'P_UITKMINAOW']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ["AFS_OPRIT", "AFS_TRNOVS", "AFS_TREINS",'AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']
    elif version == 9:
        demographic_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'P_UITKMINAOW',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'P_UITKMINAOW']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ["AFS_OPRIT", "AFS_TRNOVS", "AFS_TREINS",'AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']
    elif version == 10 or version == 13:
        demographic_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A', 'GEM_HH_GR', 'P_UITKMINAOW',
           'P_LINK_HH', 'P_HINK_HH', 'log_median_inc']
        demographic_general_names = ['AANTAL_HH', 'P_MAN', 'P_VROUW', 'GEM_HH_GR', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A']
        demographic_income_names = [ 'log_median_inc',
           'P_LINK_HH', 'P_HINK_HH', 'P_UITKMINAOW']
        demographic_age_names = [ 'P_INW_014', 'P_INW_1524', 'P_INW_2544',
           'P_INW_4564', 'P_INW_65PL']
        neighborhood_names = ["AFS_OPRIT", "AFS_TRNOVS", "AFS_TREINS",'AV1_FOOD', 'AV3_FOOD', 'AV5_FOOD', 'OAD']


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


def main_exploratory_analysis(version):
    """
    Main for some exploratory analysis on the obtained clusters
    """
    best_method = "SOM + k-means"
    pc4_labels = pd.read_csv('Results/pc4_best_labels_version_' + str(version) + "_" + best_method + '.csv')
    pc4_labels["labels"] = pc4_labels["labels"] + 1

    data_restaurants = UFS_Universe_NLnew()
    data_restaurants = data_restaurants[['pc4', 'closed']]

    data_restaurants['pc4'] = data_restaurants['pc4'].fillna(-1)
    data_restaurants['pc4'] = data_restaurants['pc4'].astype(int)
    data_restaurants['pc4'] = data_restaurants['pc4'].replace(-1, np.nan)
    data_restaurants2 = data_restaurants.merge(pc4_labels, left_on='pc4', right_on='pc4')
    print(data_restaurants)
    data_restaurants.dropna(inplace=True)
    print(data_restaurants)

    data_restaurants['pc4'] = data_restaurants['pc4'].astype(int)
    data_restaurants = data_restaurants.merge(pc4_labels, left_on='pc4', right_on='pc4')
    final_df = get_categorical_counts_clusters_df(data_restaurants, data_restaurants["labels"], var_names=["closed"])
    print(data_restaurants)
    print(final_df)
    data_restaurants_cluster_1 = data_restaurants2[data_restaurants2["labels"] == 1]
    data_restaurants_cluster_2 = data_restaurants2[data_restaurants2["labels"] == 2]


    print("Total observations in 'closed: %d" % len(data_restaurants2["closed"]))
    print("Total observations in 'closed' in cluster 1: %d" % len(data_restaurants_cluster_1["labels"]) )
    print("Total observations in 'closed' in cluster 2: %d" % len(data_restaurants_cluster_2["labels"]) )
    print("Total missing values in 'closed': %d" % data_restaurants2["closed"].isnull().sum())
    print("Total missing values in 'closed' in cluster 1: %d" % data_restaurants_cluster_1["closed"].isnull().sum())
    print("Total missing values in 'closed' in cluster 2: %d" % data_restaurants_cluster_2["closed"].isnull().sum())

    #make_latex_table_MultiIndex(final_df)
    #make_latex_table(final_df)

def main_exploratory_analysis2():
    df = pd.read_csv("Data/WX.csv")
    N = len(df)
    Y = pd.read_csv("Data/Y.csv", header=None)
    df["Y"] = Y.iloc[:, 1]

    GlobalChannel = []
    for i in range(N):
        if df.loc[i, "globalChannel_fastfood"] == 1:
            GlobalChannel.append("fastfood")
        elif df.loc[i, "globalChannel_other"] == 1:
            GlobalChannel.append("other")
        else:
            GlobalChannel.append("no dining")
    df["GlobalChannel"] = GlobalChannel
    final_df = get_categorical_counts_clusters_df(df, df["labels"], var_names=["GlobalChannel", "Y"])
    print(df)
    print(final_df)
    make_latex_table_MultiIndex(final_df, decimals=5)

def main_exploratory_analysis_complete():
    df1 = pd.read_csv("Data/WX_train_complete.csv")

    Y1 = pd.read_csv("Data/Y_train_complete.csv", header=None)
    #df1["Y"] = Y1.iloc[:, 1]
    df2 = pd.read_csv("Data/WX_test_complete.csv")

    Y2 = pd.read_csv("Data/Y_test_complete.csv", header=None)
    #df2["Y"] = Y2.iloc[:, 1]
    df = pd.concat([df1, df2], axis=0)
    Y = pd.concat([Y1, Y2], axis=0)
    df["Y"] = Y.iloc[:,1]
    GlobalChannel = []
    N = len(df)
    df = df.reset_index()
    print(N)
    print(df)
    for i in range(N):
        if df.loc[i, "globalChannel_fastfood"] == 1:
            GlobalChannel.append("Fastfood")
        elif df.loc[i, "globalChannel_no_dining"] == 1:
            GlobalChannel.append("No dining")
        elif df.loc[i, "globalChannel_dining"] == 1:
            GlobalChannel.append("Dining")
        else:
             GlobalChannel.append("Other")
    df["GlobalChannel"] = GlobalChannel
    final_df = get_categorical_counts_clusters_df(df, df["labels"], var_names=["GlobalChannel", "Y"])

    print(final_df)
    make_latex_table_MultiIndex(final_df, decimals=5)

def main_exploratory_analysis_without_outliers():
    df1 = pd.read_csv("Data/WX_train.csv")

    Y1 = pd.read_csv("Data/Y_train.csv", header=None)
    #df1["Y"] = Y1.iloc[:, 1]
    df2 = pd.read_csv("Data/WX_test.csv")

    Y2 = pd.read_csv("Data/Y_test.csv", header=None)
    #df2["Y"] = Y2.iloc[:, 1]
    df = pd.concat([df1, df2], axis=0)
    Y = pd.concat([Y1, Y2], axis=0)
    df["Y"] = Y.iloc[:,1]
    GlobalChannel = []
    N = len(df)
    df = df.reset_index()
    print(N)
    print(df)
    for i in range(N):
        if df.loc[i, "globalChannel_fastfood"] == 1:
            GlobalChannel.append("Fastfood")
        elif df.loc[i, "globalChannel_no_dining"] == 1:
            GlobalChannel.append("No dining")
        elif df.loc[i, "globalChannel_dining"] == 1:
            GlobalChannel.append("Dining")
        else:
             GlobalChannel.append("Other")
    df["GlobalChannel"] = GlobalChannel
    final_df = get_categorical_counts_clusters_df(df, df["labels"], var_names=["GlobalChannel", "Y"])

    print(final_df)
    make_latex_table_MultiIndex(final_df, decimals=5)

if __name__ == '__main__':
    version = 10

    # main_train(version)

    #main_results_best_model(version)
    #main_exploratory_analysis2()
    main_exploratory_analysis_without_outliers()
    main_exploratory_analysis_complete()
    #main_statistics(version)

