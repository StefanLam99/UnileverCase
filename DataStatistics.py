import numpy as np
import pandas as pd
from DataSets import UFS_Universe_NL, UFS_Universe_NL_ratings, Neighborhood_Descriptives, zipcode_data_2017, zipcode_data_2019
import matplotlib.pyplot as plt


def get_class_counts(X):
    '''
    Get the counts of the classes and the corresponding counts from one-dimensional arry X, in ascending order
    '''

    classes = set(X) # all different types in X
    dict_classes = dict.fromkeys(classes, 0) # make a dictionary to keep count of the number of classes
    k = len(classes) # number of types
    n = len(X) # number of observations
    # counting the types:
    for x in X:
        dict_classes[x] += 1

    # sorting the keys and values, in increasing order respective of the values
    sorted_indices = np.argsort(np.array(list(dict_classes.values())))
    sorted_classes = np.array(list(dict_classes.keys()))[sorted_indices]
    sorted_values = np.array(list(dict_classes.values()))[sorted_indices]

    return sorted_classes, sorted_values


def show_statistics_classes(X, var_name = ''):
    '''
    Gets some basic statistics of a variable with different types/classes,
    where X is an one-dimensional array
    '''

    sorted_classes, sorted_values = get_class_counts(X)
    n = len(X)
    print('Statistics of %s:' % (var_name))

    # printing statistics
    max_len = len(max(X, key=len)) # max len of the strings in X
    for cl, val in zip(sorted_classes, sorted_values):
        print('%-*s: %d (%1.1f%%)' % (max_len+2, cl, val, (float(val)/n)*100))

    print('Total number of observations: ' + str(n))
    print('\n')


def pie_plot(X, name = '', cmap= 'tab20'):
    '''
    Method to make a pie plot from an one-dimensional array X, consisting of k different classes
    Note: should not use when k is really large, such as 20
    '''

    sorted_classes, sorted_values = get_class_counts(X)
    k = len(sorted_classes) # number of types
    n = len(X) # number of observations

    # plotting the piechart
    theme = plt.get_cmap(cmap)# specifying colormap for the pie chart
    plt.gca().set_prop_cycle("color", [theme(1.*i/k) for i in range(k)])

    patches, texts, _ = plt.pie(x=sorted_values, autopct='%1.1f%%')
    plt.gca().axis('equal')
    plt.legend(patches,
               ['%s: %1.1f%%' % (cl, (float(val)/n)*100) for cl, val in zip(sorted_classes, sorted_values)],
               loc= "center right", bbox_to_anchor=(1,0.5), bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.6)
    plt.title(name +'\n observations = ' + str(n))
    plt.show()


# ToDO: add more numerical statistics
def get_numerical_statistics(X):
    '''
    Method to get the numerical statistics of an one-dimensional array X with floating numbers
    '''

    return {'mean': np.mean(X), 'std': np.std(X), 'max': np.max(X), 'min': np.min(X), 'observations': len(X)}


# ToDO: currently only uses Neighborhood_desscriptives data
def get_numerical_statistics_clusters(X, clusters):
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
        dict_cluster_stats[key] = get_numerical_statistics(dict_clusters[key])

    return dict_cluster_stats


def show_numerical_statistics_clusters(X, clusters):
    '''
    Method to print the statistics from get_numerical_statistics_clusters(X, clusters)
    '''
    dict_cluster_stats = get_numerical_statistics_clusters(X, clusters)

    for cluster in dict_cluster_stats.keys():
        print('Cluster %s: %s' % (cluster, str(dict_cluster_stats[cluster])))
    print('Total observations %d' % len(X))
    print('\n')


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
    show_statistics_classes(data_restaurants['cuisineType'], var_name='cuisineType')


    '''
    Some statistics about the postal codes: (I just did some examples)
    '''
    data_postal_codes = Neighborhood_Descriptives()
    # get the statistics on AANTAL_HH, where clusters correspond to different provinces
    show_numerical_statistics_clusters(X=data_postal_codes['AANTAL_HH'], clusters=data_postal_codes['NUTS1NAME'])
    # get the statistics on AANTAL_HH, where clusters correspond to different area_type
    show_numerical_statistics_clusters(X=data_postal_codes['AANTAL_HH'], clusters=data_postal_codes['area_type'])










