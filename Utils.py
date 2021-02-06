import numpy as np
import sys

def euclidean_distance(x, y):
    """
    Calculates the euclidean distance betwewen x and y
    """
    return np.sqrt(np.sum((x-y)**2))


def hit_rate(corr, preds):
    """
    Calculates the hit rate between two arrays: corr and preds.
    """
    return np.mean(corr == preds)


def infer_cluster_labels(pred_clusters, actual_labels):
    """
    Method that infers the corresponding label to each cluster, if the
    actual labels are available.
    :param pred_clusters: the predicted clusters of the training data
    :param actual_labels: the actual labels of the training data
    :return: a dictionary with as key the cluster and value the coresponding label
    """
    inferred_labels = {}
    n_clusters = len(set(actual_labels))

    for i in range(n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(pred_clusters == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        # print(labels)
        # print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels


def infer_data_labels(pred_clusters, cluster_labels):
    """
    Method to infers the ACTUAL label for each observation, given the predicted cluster.
    :param pred_clusters: the predicted clusters of a data sample
    :param cluster_labels:  the predicted clusters of a data sample
    :return: the ACTUAL predicted labels of a data sample
    """
    predicted_labels = np.zeros(len(pred_clusters)).astype(np.uint8)

    for i, cluster in enumerate(pred_clusters):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels

def normalize(df):
    """
    Normalizes a dataframe to be in the range [0,1], without altering the distribution, using a min-max scale.
    :param df: dataframe containing the data
    :retrun normalized dataframe
    """
    all_df = df.copy()
    df.dropna(inplace=True)
    max = np.max(df, axis=0)
    min = np.min(df, axis=0)
    all_df = (all_df-min)/(max-min)
    return all_df, max.values, min.values


def make_dir(file_path):
    '''
    Makes a directory for a file path if it does not already exist
    '''
    split = file_path.rsplit('/',1)
    dir = split[0]
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)



