import pandas as pd
import numpy as np
from DataSets import zipcode_data_2019, zipcode_data_2017
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from Utils import normalise
import matplotlib.pyplot as plt
import seaborn as sns
from DataStatistics import  get_numerical_statistics_clusters_df
from impyute.imputation.cs import mice, fast_knn
from scipy.stats import jarque_bera
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break
#pd.set_option('display.max_rows', None)
from time import time


def main_statistics():
    '''
    Main to show the distribtuions of different variables
    '''
    version = 4
    data = pd.read_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv")
    version = 5
    data2 = pd.read_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv")



    data_2019 = zipcode_data_2019().replace(-99997, np.nan)
    #data_2019.dropna(inplace=True)
    data = data.merge(data_2019[["pc4", "AANTAL_HH"]], left_on='pc4', right_on='pc4')
    data.dropna(inplace=True)

    x = data["INWONER"]
    sns.distplot(x, hist=True, bins=30, kde=True, color="darkblue", hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.show()

    x = data["INWONER"]/data["AANTAL_HH"]
    sns.distplot(x, hist=True, bins=30, kde=True, color="darkblue", hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.show()

    x = data["UITKMINAOW"]
    sns.distplot(x, hist=True, bins=30, kde=True, color="darkblue", hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.show()

    x = data["UITKMINAOW"]/data["AANTAL_HH"]
    sns.distplot(x, hist=True, bins=30, kde=True, color="darkblue", hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.show()

    for var in data2.columns:
        x = data2[var]

        x.dropna(inplace=True)
        JB = jarque_bera(x)
        print("%s has a JB-statistics of %.3f with p-value = %.3f" % (var, JB[0], JB[1]))
        print("observations: " + str(len(x)))
        print("")


def main_preprocessing_version_4():
    """
    Main to perform some data preprocessing of the data
    """

    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()
    data_v3 = pd.read_csv("Data/zipcodedata_version_3_nanincluded.csv")
    final_data = pd.DataFrame(data_2019["pc4"], columns=["pc4"])  # initialize it so we have the # of rows beforehand

    # gender distributions
    gender_data = pd.DataFrame(data_2019[["MAN", "VROUW"]], columns=["MAN", "VROUW"])
    gender_data = gender_data.replace(-99997, np.nan)  # replace weird values with NaN
    gender_data.dropna(inplace=True)
    total_gender = gender_data["MAN"] + gender_data["VROUW"]
    gender_data["P_MAN"] = gender_data["MAN"]/total_gender
    gender_data["P_VROUW"] = gender_data["VROUW"]/total_gender
    final_data["INWONER"] = data_v3["INWONER"]
    final_data["P_MAN"] = gender_data["P_MAN"]
    final_data["P_VROUW"] = gender_data["P_VROUW"]


    # age distributions
    age_data = pd.DataFrame(data_v3[["INW_014", "INW_1524", "INW_2544", "INW_4564", "INW_65PL"]])
    age_data.dropna(inplace=True)
    total_ages = age_data["INW_014"] + age_data["INW_1524"] + age_data["INW_2544"] + age_data["INW_4564"] + age_data["INW_65PL"]
    final_data["P_INW_014"] = age_data["INW_014"]/total_ages
    final_data["P_INW_1524"] = age_data["INW_1524"]/total_ages
    final_data["P_INW_2544"] = age_data["INW_2544"]/total_ages
    final_data["P_INW_4564"] = age_data["INW_4564"]/total_ages
    final_data["P_INW_65PL"] = age_data["INW_65PL"]/total_ages
    #data["P_AGES"] =  data["Page_INW_014"] + data["P_INW_1524"] +     data["P_INW_2544"] +     data["P_INW_4564"] +     data["P_INW_65PL"]


    # sum of food facilities in a 1, 3, 5 km radius
    food_data = pd.DataFrame(data_2017[["AV1_CAFE", "AV1_CAFTAR", "AV1_RESTAU", "AV3_CAFE", "AV3_CAFTAR", "AV3_RESTAU", "AV5_CAFE", "AV5_CAFTAR", "AV5_HOTEL", "AV5_RESTAU"]])
    food_data = food_data.replace(-99997, np.nan)  # replace weird values with NaN
    food_data.dropna()
    final_data["AV1_FOOD"] = food_data["AV1_CAFE"] + food_data["AV1_CAFTAR"] + food_data["AV1_RESTAU"]
    final_data["AV3_FOOD"] = food_data["AV3_CAFE"] + food_data["AV3_CAFTAR"] + food_data["AV3_RESTAU"]
    final_data["AV5_FOOD"] = food_data["AV5_CAFE"] + food_data["AV5_CAFTAR"] + food_data["AV5_RESTAU"]


    # other relevant variables
    variable_names = ["OAD", "P_NL_ACHTG", "P_WE_MIG_A", "P_NW_MIG_A", "GEM_HH_GR", "UITKMINAOW", "P_LINK_HH",
                      "P_HINK_HH", "median_inc"]

    for name in variable_names:
        final_data[name] = data_v3[name]
    final_data["log_median_inc"] = np.log(final_data["median_inc"])
    final_data = final_data.drop(columns= ["median_inc"])  # only include log median income

    # save data
    version = 4  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)
    print(final_data)


def main_preprocessing_version_5():
    """
    Main to perform some data preprocessing of the data
    """
    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()
    data_v4 = pd.read_csv("Data/zipcodedata_version_4_nanincluded.csv")


    # process the data....
    AANTAL_HH = data_2019["AANTAL_HH"]
    AANTAL_HH = AANTAL_HH.replace(-99997, np.nan)
    data_v4.insert(1, "INWONER_HH", data_v4["INWONER"]/AANTAL_HH)  # number of inhabitants per household
    data_v4 = data_v4.drop(["INWONER"], axis=1)

    data_v4.insert(18, "UITKMINAOW_HH", data_v4["UITKMINAOW"]/AANTAL_HH)  # number of inhabitants receiving social benefits per household
    data_v4 = data_v4.drop(["UITKMINAOW"], axis=1)
    final_data = data_v4

    # save data
    version = 5  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)
    print(final_data)


def main_preprocessing_version_6():
    """
    Main to perform some data preprocessing of the data
    """
    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()
    data = pd.read_csv("Data/zipcodedata_version_5_nanincluded.csv")
    data = data.drop(["INWONER_HH"], axis=1)
    final_data = data

    # process the data....
    AANTAL_HH = data_2019["AANTAL_HH"]
    AANTAL_HH = AANTAL_HH.replace(-99997, np.nan)

    data.insert(1, "AANTAL_HH", AANTAL_HH)
    # save data
    version = 6  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)
    print(final_data)


def main_preprocessing_version_7():
    """
    Main to perform some data preprocessing of the data
    """
    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()
    data = pd.read_csv("Data/zipcodedata_version_6_nanincluded.csv")
    final_data = data

    # process the data....
    UITKMINAOW = data_2019["UITKMINAOW"]
    UITKMINAOW = UITKMINAOW.replace(-99997, np.nan)
    INWONER = data_2019["INWONER"]
    INWONER = INWONER.replace(-99997, np.nan)
    data.insert(18, "P_UITKMINAOW", UITKMINAOW/INWONER)
    data = data.drop(["UITKMINAOW_HH"], axis=1)
    final_data = data

    # save data
    version = 7  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)
    print(final_data)


def main_preprocessing_version_8():
    """
    Main to perform some data preprocessing of the data
    """
    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()
    data = pd.read_csv("Data/zipcodedata_version_7_nanincluded.csv")

    # process the data....
    AFS_OPRIT = data_2017["AFS_OPRIT"]
    AFS_OPRIT = AFS_OPRIT.replace(-99997, np.nan)
    AFS_TRNOVS = data_2017["AFS_TRNOVS"]
    AFS_TRNOVS = AFS_TRNOVS.replace(-99997, np.nan)
    AFS_TREINS = data_2017["AFS_TREINS"]
    AFS_TREINS = AFS_TREINS.replace(-99997, np.nan)
    data.insert(8, "AFS_OPRIT", AFS_OPRIT)
    data.insert(8, "AFS_TRNOVS", AFS_TRNOVS)
    data.insert(8, "AFS_TREINS", AFS_TREINS)
    final_data = data


    # save data
    version = 8  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)
    print(final_data)


def main_preprocessing_version_9():
    """
    Main to perform some data preprocessing of the data
    """
    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()
    data = pd.read_csv("Data/zipcodedata_version_8_nanincluded.csv")


    # process the data....
    #data = data.drop(["log_median_inc"], axis=1)
    final_data = data

    # save data
    version = 9  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)
    print(final_data)


def main_preprocessing_version_10():
    """
    Main to perform some data preprocessing of the data
    """
    # Read the data with the NaNs included:
    data_2017 = zipcode_data_2017()
    data_2019 = zipcode_data_2019()

    # process the data...
    data = pd.read_csv("Data/zipcodedata_version_9_nanincluded.csv")
    data["P_MAN"] = data["P_MAN"]*100
    data["P_VROUW"] = data["P_VROUW"]*100
    data["P_INW_014"] = data["P_INW_014"]*100
    data["P_INW_1524"] = data["P_INW_1524"]*100
    data["P_INW_2544"] = data["P_INW_2544"]*100
    data["P_INW_4564"] = data["P_INW_4564"]*100
    data["P_INW_65PL"] = data["P_INW_65PL"]*100
    data["P_UITKMINAOW"] = data["P_UITKMINAOW"]*100

    final_data = data
    # save data
    version = 10  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv", index=False)

    print(final_data)


def main_imputation(version):
    '''
    Main to normalize/impute the data.
    :return:
    '''
    # Read the data with the NaNs included:
    data = pd.read_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv")
    pc4 = data['pc4']
    X = data.iloc[:,1:]  # without pc4 column
    n = X.shape[0]
    column_names = data.columns


    #  imputing the data with knn
    X_normalized, max, min = normalise(X)  # first normalize since we use euclidean distance in knn
    imputed_X_normalized = KNNImputer(n_neighbors=20).fit_transform(X_normalized)
    imputed_X = imputed_X_normalized * (max-min) + min  # rescale back to original values
    imputed_X_normalized = np.concatenate((pc4.values.reshape((n, 1)), imputed_X_normalized), axis=1)
    imputed_X = np.concatenate((pc4.values.reshape((n, 1)), imputed_X), axis=1)


    ''' 
    #  removing outliers
    X = imputed_X_normalized
    clf = IsolationForest(max_samples=100, random_state=0, contamination=0.05)
    LOF = LocalOutlierFactor()
    print(X)
    outlier_detection = DBSCAN(min_samples=2, eps=3)
    preds = outlier_detection.fit_predict(X)
    #preds = clf.fit_predict(X)
    #preds = LOF.fit_predict(X)
    print(len(preds))
    print(np.sum(preds==-1))
    print(np.sum(preds==1))

    imputed_X = imputed_X[preds!=-1]
    imputed_X_normalized = imputed_X_normalized[preds!=-1]
    '''

    imputed_data = pd.DataFrame(imputed_X, columns=column_names)
    imputed_data_normalized = pd.DataFrame(imputed_X_normalized, columns = column_names)

    imputed_data.to_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv", index=False)
    imputed_data_normalized.to_csv("Data/zipcodedata_KNN_normalized_version_" + str(version) + ".csv", index=False)
    # show the description of the imputed data
    print(imputed_data)
    for name in column_names:
        print(imputed_data[name].describe())



if __name__ == '__main__':
    version = 9
    #main_preprocessing_version_4()
    #main_preprocessing_version_5()
    #main_preprocessing_version_6()
    #main_preprocessing_version_7()
    #main_preprocessing_version_8()
    #main_preprocessing_version_9()
    main_preprocessing_version_10()

    main_imputation(version)
    #main_statistics()




