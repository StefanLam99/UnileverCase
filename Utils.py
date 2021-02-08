import numpy as np
import sys
from DataSets import Neighborhood_Descriptives
from DataStatistics import get_categorical_counts_clusters_df
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


def make_latex_table(df, scale=0.6):
    """
      Make a latex table from a dataframe
    """
    n_cols = len(df.columns) + 2

    # write the first row of the dataframe
    first_row = ""
    for i, col_name in enumerate(np.array(df.columns)):
        if i ==0:
            first_row += "&&" + col_name
        else:
            first_row += "&" + col_name
    first_row += "\\\\ \hline"

    # write the data of the dataframe
    data = ""
    for i, row_name in enumerate(np.array(df.index)):
        data += str(row_name)
        for j, element in enumerate(df.loc[row_name]):
            if j==0:
                data += "&& %s" % element
            else:
                data += "& %s" % element
        # last row do not end the line:
        if i < len(np.array(df.index))-1:
            data += "\\\\ \n"
        else:
            data += "\\\\"
    data += "\hline \hline"

    # print the latex table
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Look at this table :D}")
    print("\\label{tab: EnterYourLabel}")
    print("\\scalebox{%.2f}{" % scale)
    print("\\begin{tabular}{" + "l"*n_cols + "} \hline \hline")
    print(first_row)
    print(data)
    print("\\end{tabular}}")
    print("\\end{table}")

def make_latex_table_MultiIndex(df, scale=0.6):
    """
    Make a latex table from a MultiIndex df with two levels/
    """
    n_cols = len(df.columns.codes[0]) +len(df.columns.levels[0]) + 1
    print(n_cols)
    levels = df.columns.levels
    codes = df.columns.codes
    names = df.columns.names

    # write the first row of the multiIndex dataframe
    (uniques, indices, counts) = np.unique(np.array(codes[0]), return_index=True, return_counts=True)
    sorted_indices= np.argsort(indices)
    sorted_counts = counts[sorted_indices]
    sorted_uniques = uniques[sorted_indices]
    first_row = "  "
    c_lines = ""
    c_start = 3
    for count, unique in zip(sorted_counts, sorted_uniques):
        first_row += "&& \multicolumn{%d}{c}{%s} " % (count, np.array(levels[0])[unique])
        c_end = c_start + count - 1
        c_lines += "\\cline{%d-%d} " % (c_start, c_end)
        c_start = c_end + 2
    first_row += "\\\\" + c_lines

    # write the second row of the multiIndex dataframe
    second_row = ""
    current_ncols = 0  # keep track of the current number of columns of the current variable
    current_index = 0  # keep track of the first index of the current variable

    for i, code in enumerate(codes[1]):
        if(i == current_ncols):
            second_row += " && %s " % (levels[1][code])
            current_ncols += sorted_counts[current_index]
            current_index+=1
        else:
            second_row += " & %s " % (levels[1][code])
    second_row += "\\\\ \hline"


    # write the data of the multiIndex dataframe
    data = ""

    for i, row_name in enumerate(df.index):
        current_ncols = 0  # keep track of the current number of columns of the current variable
        current_index = 0  # keep track of the first index of the current variable
        data += row_name
        row = np.array(df.loc[row_name])
        for j, element in enumerate(row):
            if (j == current_ncols):
                data += " && %s " % element
                current_ncols += sorted_counts[current_index]
                current_index += 1
            else:
                data += " & %s " % element

        # last row do not end the line:
        if(i<len(df.index)-1):
            data += "\\\\ \n"
        else:
            data += "\\\\"
    data += "\hline \hline"



    # print the latex table
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Look at this table :D}")
    print("\\label{tab: EnterYourLabel}")
    print("\\scalebox{%.2f}{" % scale)
    print("\\begin{tabular}{" + "l"*n_cols + "} \hline \hline")
    print(first_row)
    print(second_row)
    print(data)
    print("\\end{tabular}}")
    print("\\end{table}")


if __name__ == '__main__':
    neighbor_data = Neighborhood_Descriptives()
    df_counts = get_categorical_counts_clusters_df(neighbor_data, np.array(neighbor_data["NUTS2NAME"]), var_names=["DEGURBA", "higher_education", "COASTAL_AREA_yes_no"])


    print(df_counts)

    print(df_counts.columns)
    print(df_counts.columns.levels)
    print(neighbor_data.columns[0])

    print("\\")
    (unique, indices, counts) = np.unique(list(df_counts.columns.codes[0]), return_index=True, return_counts=True)
    print(unique[0])
    print(np.array(df_counts.columns.codes[0]))
    print(set(np.array(df_counts.columns.codes[0])))
    print(unique)
    print(indices)
    print(counts)
    print(df_counts.index[0])
    a = df_counts.index[0]
    print(np.array(df_counts.loc[a]))
    print(neighbor_data.columns)
    print(neighbor_data.index)
    make_latex_table(neighbor_data.iloc[1:5,1:5])
    make_latex_table_MultiIndex(df_counts)






