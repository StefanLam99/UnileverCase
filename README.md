# Description of the code used for the cluster analysis

## Clustering Methods

**SOM.py**:  Class to create an object which implements a SOM network as described by (Kohonen 1990). It can be used to train a SOM network and for predictions of cluster labels.

**two_stage_clustering.py**: Class to create an object which implements the two-stage clustering procedure as described in this paper. The first stage is a SOM network and the second stage is either k-means or GMM. It can be used to train the two-stage clustering model and for prediction of the cluster labels.

## Utility Classes

**DataSets.py**:  Class which loads several data sets from the  CBS and UFS to dataframes.

**DataStatistics.py**: Class which has functions to obtain statistics from several kinds of dataframes.

**Utils.py**: Class which contains simple functions used for the implemented clustering methods.

## Main Classes

**main_data_preprocessing,py**:  Main to preprocess the regional data set from the CBS.

**main_cluster_validation.py**:  Main used to determine the optimal $k$ for each method using our cluster validation measures.

**main_clustering.py**: Main to obtain the labels, plots and statistics of the implemented models with their optimal k determined from main_cluster_validation.py.

**main_geopandas.py**:  Main to obtain the map of the Netherlands with as data points the restaurants corresponding to the label from the best clustering method.


# Description of the code used for the GLH models

## GLH Models

**multilog.stan**: Stan model code of the Pooled model.

**su_model.stan**: Stan model code of the SU model.

## Main Classes

**main_data_preprocessing_GLH.ipynb**: Main to preprocess the restaurant data set from Unilever and merge it with the preprocessed zip code data set and the labels found in the clustering step.

**main_stan_GLH.R**: Main to run all the GLH models, obtain the predictions and create the confusion matrices.







