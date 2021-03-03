import numpy as np
import pandas as pd
from DataSets import UFS_Universe_NL, Neighborhood_Descriptives
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from Utils import make_dir
from time import time
from scipy import stats
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break


def get_class_counts(X):
    '''
    Get the counts of the classes and the corresponding counts from one-dimensional arry X, in ascending order
    '''
    return X.value_counts(ascending=True, dropna = False)


def show_statistics_classes(X, var_name = ''):
    '''
    Gets some basic statistics of a variable with different types/classes,
    where X is an one-dimensional array
    '''

    class_counts = get_class_counts(X)
    n = len(X)
    print('Statistics of %s:' % (var_name))

    # printing statistics
    class_counts.index = class_counts.index.fillna('NaN')
    max_len = len(max(class_counts.index, key=len))  # max len of the strings in X
    for class_name, class_val in class_counts.items():
        print('%-*s: %d (%1.1f%%)' % (max_len + 2, class_name, class_val, (float(class_val) / n) * 100))

    print('Total number of observations: ' + str(n))
    print('\n')


def pie_plot(X, name = '', cmap= 'tab20'):
    '''
    Method to make a pie plot from an one-dimensional array X, consisting of k different classes
    Note: should not use when k is really large, such as 20
    '''

    class_counts = get_class_counts(X)
    sorted_classes, sorted_values = class_counts.index, class_counts
    k = len(sorted_classes) # number of types
    n = len(X) # number of observations

    # plotting the piechart
    theme = plt.get_cmap(cmap)# specifying colormap for the pie chart
    plt.gca().set_prop_cycle("color", [theme(1.*i/k) for i in range(k)])

    patches, texts, _ = plt.pie(x=sorted_values, autopct='%1.1f%%')
    plt.gca().axis('equal')
    plt.legend(patches,
               ['%s: %1.1f%%' % (cl, (float(val)/n)*100) for cl, val in zip(sorted_classes, sorted_values)],
               loc= "center right", bbox_to_anchor=(1, 0.5), bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.6)
    plt.title(name +'\n Observations = ' + str(n))
    plt.show()


# ToDO: add more numerical statistics
def get_numerical_statistics(X, round=3):
    '''
    Method to get the numerical statistics of an one-dimensional array X with floating numbers
    '''
    orig_n_obs = len(X)  # number of observations including missing values
    X = np.array(X)
    X = X[np.logical_not(np.isnan(X))]
    return {'mean': np.round(np.mean(X), round), 'std': np.round(np.std(X), round), 'median': np.round(np.median(X), round)
        , 'max': np.round(np.max(X), round), 'min': np.round(np.min(X), round), 'Observations': orig_n_obs, 'Jarque-Bera Test': stats.jarque_bera(X)[0]}


def get_numerical_statistics_clusters(X, clusters, round=3):
    '''
    Method to get the numerical statistics of each cluster in an one-dimensional array X with floats,
    where a cluster corresponds to certain pc4 codes. Returns a dictionary....
    NOTE: X and clusters are both arrays with the same length and same order.
    '''

    # make a dictionary to store the values of X in the corresponding cluster:
    dict_clusters = {}
    for cluster in set(clusters): # a bit tedious, but I see no easier option atm...
        dict_clusters[cluster] = []

    for x, cluster in zip(X, clusters):
        dict_clusters[cluster] += [x]

    # make a dictionary which will get all the statistics of a cluster:
    dict_cluster_stats = {}
    for key in dict_clusters.keys():
        dict_cluster_stats[key] = get_numerical_statistics(dict_clusters[key], round=3)

    return dict_cluster_stats


def get_numerical_statistics_clusters_df(df, clusters, var_names, stat_names=["mean", "std", "max", "min"], round=3):
    """
    Method to obtain the statistics of the clusters
    :param df: dataframe with the data
    :param clusters: array with the corresponding cluster for each observation in df
    :param var_names: the names of the variables to put in the dataframe
    :param stat_names: the statistics we want, choice: "mean". "std", "max", "min, "observations
    :param round: the number of decimals to round
    :return: a MultiIndex dataframe
    """
    #  determine number of clusters, variables, statistics and the size of the data
    n_clusters = len(set(clusters))
    n_vars = len(var_names)
    n_stats = len(stat_names)
    result = np.zeros((n_clusters + 1, n_vars * n_stats))

    #  Make the indices for the row and columns of the dataframe
    dict_clusters = get_numerical_statistics_clusters(df[var_names[0]], clusters)
    indices_row = ["cluster " + str(e) for e in dict_clusters.keys()]
    indices_row += ["total"]
    #indices_row = ["cluster " + str(int(i)) for i in range(n_clusters)]
    iterables =[var_names, stat_names]
    indices_col = pd.MultiIndex.from_product(iterables, names=["variable", "statistic"])

    # get the values for each cluster
    observations = np.zeros(n_clusters+1)
    for i, var in enumerate(var_names):
        dict_clusters = get_numerical_statistics_clusters(df[var], clusters, round=3)
        for j, cluster in enumerate(dict_clusters.keys()):  # might not be sorted, so do it in this way
            observations[j] = dict_clusters[cluster]["Observations"]
            for k, stat in enumerate(stat_names):
                result[j, i*n_stats+k] = dict_clusters[cluster][stat]

    # get the values for all cluster together
    for i, var in enumerate(var_names):
        dict = get_numerical_statistics(df[var], round=3)
        for k, stat in enumerate(stat_names):
            result[n_clusters, i*n_stats+k] = dict[stat]
    observations[n_clusters] = np.shape(df)[0]

    # make the dataframe
    result_df = pd.DataFrame(result, index=indices_row, columns=indices_col)
    result_df.insert(0, "Observations", observations)
    #result_df["observations"] = observations  # add observations to the second level of the last column
    return result_df.round(round)


def get_numerical_statistics_clusters_df_inversed(df, clusters, dict_var_names, stat_names=["mean", "std", "max", "min"], round=3):
    """
    Method to obtain the statistics of the clusters
    :param df: dataframe with the data
    :param clusters: array with the corresponding cluster for each observation in df
    :param dict_var_names: keys are variable type, values are the variable names. (to be put in the df)
    :param stat_names: the statistics we want, choice: "mean". "std", "max", "min, "observations
    :param round: the number of decimals to round
    :return: a MultiIndex dataframe
    """
    #  determine number of clusters, variables, statistics and the size of the data
    n_clusters = len(set(clusters))
    n_groups = len(dict_var_names)
    n_vars = np.sum([len(x) for x in dict_var_names.values()])
    n_stats = len(stat_names)
    result = np.zeros((n_vars + n_groups+1, (n_clusters + 1)*n_stats))

    #  Make the indices for the row and columns of the dataframe
    indices_row = []
    for key in dict_var_names.keys():
        indices_row += [key]
        for var in dict_var_names[key]:
            indices_row += [var]
    indices_row += ["Observations"]


    columns = ["cluster " + str(int(e)) for e in range(n_clusters)]
    columns += ["Overall"]
    #indices_row = ["cluster " + str(int(i)) for i in range(n_clusters)]
    iterables =[columns, stat_names]
    indices_col = pd.MultiIndex.from_product(iterables, names=["variable", "statistic"])

    # get the values for each cluster
    observations = np.zeros(n_clusters+1)
    i = 0  # keep track of current row
    for l, key in enumerate(dict_var_names):
        i += 1  # skip the group name row
        for var in dict_var_names[key]:
            dict_clusters = get_numerical_statistics_clusters(df[var], clusters, round=3)
            for j, cluster in enumerate(dict_clusters.keys()):  # might not be sorted, so do it in this way
                observations[j] = dict_clusters[cluster]["Observations"]
                for k, stat in enumerate(stat_names):
                    result[i, j*n_stats+k] = dict_clusters[cluster][stat]
            i+=1

    # get the values for all cluster together
    i = 0  # keep track of current row
    for l, key in enumerate(dict_var_names):
        i+=1  # skip the group name row
        for var in dict_var_names[key]:
            dict = get_numerical_statistics(df[var], round=3)
            for k, stat in enumerate(stat_names):
                result[i, n_clusters*n_stats + k] = dict[stat]
            i+=1
    observations[n_clusters] = np.shape(df)[0]

    # add observation row
    for i, observation in enumerate(observations):
        result[n_groups + n_vars, i*n_stats] = observation

    result_df = pd.DataFrame(result, index=indices_row, columns=indices_col)
    return result_df.round(round)


def show_numerical_statistics_clusters(X, clusters):
    '''
    Method to print the statistics from get_numerical_statistics_clusters(X, clusters)
    '''
    dict_cluster_stats = get_numerical_statistics_clusters(X, clusters)

    for cluster in dict_cluster_stats.keys():
        print('Cluster %s: %s' % (cluster, str(dict_cluster_stats[cluster])))
    print('Total observations %d' % len(X))
    print('\n')


def get_categorical_counts_clusters(X, clusters):
    '''
    Method to get the categorical counts of each cluster in an one-dimensional array X with floats,
    where a cluster corresponds to certain pc4 codes. Returns a dictionary....
    NOTE: X and clusters are both arrays with the same length and same order.
    '''

    # make a dictionary to store the values of X in the corresponding cluster:
    dict_clusters = {}
    classes = list(set(X))
    for cluster in set(clusters): # a bit tedious, but I see no easier option atm...
        dict_clusters[cluster] = []

    for x, cluster in zip(X, clusters):
        dict_clusters[cluster] += [x]

    # make a dictionary which will get all the counts of classes for a cluster:
    dict_cluster_stats = {}
    for key in dict_clusters.keys():
        dict_counts = {}
        for clas in classes:
            dict_counts[clas] = 0
        uniques, counts = np.unique(np.array(dict_clusters[key]), return_counts=True)
        for unique, count in zip(uniques, counts):
            dict_counts[unique] += count
        #dict_counts = dict(zip(unique, counts))
        dict_cluster_stats[key] = dict_counts

    return dict_cluster_stats


def get_categorical_counts_clusters_df(df, clusters, var_names):
    """
    Method to obtain the counts of categorical data for the clusters
    :param df: dataframe with categorical data
    :param clusters: array with the corresponding cluster for each observation in df
    :param var_names: the names of the variables to put in the dataframe
    :return: a MultiIndex dataframe
    """

    n_clusters = len(set(clusters))

    # determine column and row indices and shape of the dataframe
    first_level = []
    second_level = []
    n_classes = 0
    for var_name in var_names:
        n_classes += len(set(df[var_name]))
        first_level += len(set(df[var_name]))*[var_name]
        second_level += list(set(df[var_name]))

    # column indices
    result = np.zeros((n_clusters, n_classes))
    result_counts = np.zeros((n_clusters, n_classes))
    tuples = list(zip(first_level, second_level))
    indices_col = pd.MultiIndex.from_tuples(tuples, names=["variable", "class"])

    # row indices
    #indices_row = ["cluster " + str(int(i)) for i in range(n_clusters)]
    dict_clusters = get_categorical_counts_clusters(df[var_names[0]], clusters)
    indices_row = ["cluster " + str(e) for e in dict_clusters.keys()]
    indices_row += ["Total"]

    # get the values for the results:
    cur_len_cols = 0  # keeps track of the current position of the column
    observations = np.zeros(n_clusters)
    for i, var_name in enumerate(var_names):
        dict_clusters = get_categorical_counts_clusters(df[var_name], clusters)
        observations = np.zeros(n_clusters)  # this is repetitive, but works
        for j, cluster in enumerate(dict_clusters.keys()):  # might not be sorted, so do it in this way
            for k, class_name in enumerate(list(set(df[var_name]))):  # iterate over all classes

                result[j, k+cur_len_cols] = dict_clusters[cluster][class_name]
                result_counts[j, k + cur_len_cols] = dict_clusters[cluster][class_name]
                observations[j] += dict_clusters[cluster][class_name]
            result[j,:] = result[j, :] / observations[j]
        cur_len_cols +=len(set(df[var_name]))

    last_row = np.sum(result_counts, axis=0)
    last_row = last_row/np.sum(observations)
    result = np.vstack((result, last_row.reshape((1,-1))))
    result_df = pd.DataFrame(result, index=indices_row, columns=indices_col)
    observations = np.concatenate((observations.reshape((1,-1)), np.array(np.sum(observations)).reshape((1,-1))), axis=1)
    result_df["Observations"] = observations.reshape((-1,1))

    return result_df


def plot_clusters(X_embedded, labels, title="", save_path=None, s=1):
    """
    Visualises the observations from X_embedded with the corresponding clusters in a 2D plot
    """

    #  make the plots
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    fig.patch.set_visible(False)
    ax.axis('off')
    # scatter = ax.scatter(x=X_sorted[:, 0], y=X_sorted[:, 1], cmap="Paired", c=labels_sorted, s=1, vmin=0, vmax=6)
    scatter = ax.scatter(x=X_embedded[:, 0], y=X_embedded[:, 1], cmap="Paired", c=labels, s=s, vmin=0, vmax=6)
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), loc="upper left", title='Cluster', prop={'size': 10}, fancybox=True)

    if save_path != None:
        make_dir(save_path)
        plt.savefig(save_path)
    #fig.set_size_inches(8, 8)
    plt.show()


if __name__ == '__main__':
    '''
    Some statistics about the restaurants (some examples):
    '''

    data_restaurants = UFS_Universe_NL()
    # data_restaurants = UFS_Universe_NL_ratings() #if we have the rating

    # able to plot piecharts for the following class variables:
    pie_plot(data_restaurants['globalChannel'], name='restaurant type')
    pie_plot(data_restaurants['closed'], name='Restaurant situation')

    # print the basic statistics for the class variables with too many classes
    show_statistics_classes(data_restaurants['city'], var_name='city')
    show_statistics_classes(data_restaurants['closed'], var_name = 'closed')
    show_statistics_classes(data_restaurants['cuisineType'], var_name='cuisineType')

    '''
    Some statistics about the postal codes: (I just did some examples)
    '''
    data_postal_codes = Neighborhood_Descriptives()
    # get the statistics on AANTAL_HH, where clusters correspond to different provinces
    show_numerical_statistics_clusters(X=data_postal_codes['AANTAL_HH'], clusters=data_postal_codes['NUTS1NAME'])
    # get the statistics on AANTAL_HH, where clusters correspond to different area_type
    show_numerical_statistics_clusters(X=data_postal_codes['AANTAL_HH'], clusters=data_postal_codes['area_type'])

    '''
    Example to obtain dataframes of numerical/categorical data:
    '''
    neighbor_data = Neighborhood_Descriptives()
    data = pd.read_csv("Data/zipcodedata_version_2_nanincluded.csv")
    #  obtain some random cluster:
    np.random.seed(0)
    random_clusters = np.random.randint(0, 3, np.shape(data)[0])

    # numerical data:
    df_statistics = get_numerical_statistics_clusters_df(neighbor_data, np.array(neighbor_data["NUTS2NAME"]), var_names=['INW_014', "MAN",  'AANTAL_HH'])
    print(df_statistics)
    print('\n')

    # categorical data:
    df_counts = get_categorical_counts_clusters_df(neighbor_data, np.array(neighbor_data["NUTS2NAME"]), var_names=["DEGURBA", "higher_education", "COASTAL_AREA_yes_no"])
    print(df_counts)

    df_test = get_numerical_statistics_clusters_df(data, random_clusters, var_names=['INW_014', "MAN",  'AANTAL_HH'])
    print(df_test)













