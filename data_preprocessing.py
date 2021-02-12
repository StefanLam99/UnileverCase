import pandas as pd
import numpy as np
from DataSets import zipcode_data_2019, zipcode_data_2017
from sklearn.impute import KNNImputer
from Utils import normalise
from DataStatistics import  get_numerical_statistics_clusters_df
from impyute.imputation.cs import mice, fast_knn
from scipy import stats
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break
pd.set_option('display.max_rows', None)
from time import time


def main_preprocessing():
    """
    MAin to perform some data preprocessing of the data
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
    variable_names = [ "OAD", "P_NL_ACHTG", "P_WE_MIG_A", "P_NW_MIG_A", "GEM_HH_GR", "UITKMINAOW", "P_LINK_HH", "P_HINK_HH", "median_inc"]
    for name in variable_names:
        final_data[name] = data_v3[name]
    final_data["log_median_inc"] = np.log(final_data["median_inc"])
    final_data = final_data.drop(columns= ["median_inc"])  # only include log median income

    # save data
    version = 4  # specify version
    final_data.to_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv")
    print(final_data)


def main_imputation():
    '''
    Main to normalize/impute the data.
    :return:
    '''
    # Read the data with the NaNs included:
    version = 4  # specify version of the data
    data = pd.read_csv("Data/zipcodedata_version_" + str(version) + "_nanincluded.csv")
    pc4 = data['pc4']
    X = data.iloc[:,1:]  # without pc4 column
    n = X.shape[0]
    column_names = data.columns

    #  imputing the data with knn
    X_normalized, max, min = normalise(X)  # first normalize since we use euclidean distance in knn
    imputed_X_normalized = KNNImputer(n_neighbors=30).fit_transform(X_normalized)
    print(imputed_X_normalized.shape)
    print(min)
    print(max.shape)
    imputed_X = imputed_X_normalized * (max-min) + min  # rescale back to original values

    imputed_X_normalized = np.concatenate((pc4.values.reshape((n, 1)), imputed_X_normalized), axis=1)
    imputed_X = np.concatenate((pc4.values.reshape((n, 1)), imputed_X), axis=1)

    imputed_data = pd.DataFrame(imputed_X, columns=column_names)
    imputed_data_normalized = pd.DataFrame(imputed_X_normalized, columns = column_names)

    imputed_data.to_csv("Data/zipcodedata_KNN_version_" + str(version) + ".csv", index=False)
    imputed_data_normalized.to_csv("Data/zipcodedata_KNN_normalized_version_" + str(version) + ".csv", index=False)

    # show the description of the imputed data
    print(imputed_data)
    for name in column_names:
        print(imputed_data[name].describe())



if __name__ == '__main__':
    main_preprocessing()
    main_imputation()
    # main()




