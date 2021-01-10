import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) # makes sure that python prints all columns



def Neighborhood_Descriptives():
    """
    Returns the Neighborhood_Descriptive.xlsx file as a pd dataframe.

    Variables in order of columns:
    'LAUCODE', 'LAU_NAMENATIONAL', 'LAU_NAMELATIN', 'NUTS3CODE',
       'NUTS3NAME', 'POPULATION', 'DEGURBA', 'COASTAL_AREA_yes_no',
       'higher_education', 'CITY_ID', 'CITY_NAME', 'GREATER_CITY_ID',
       'GREATER_CITY_NAME', 'FUA_ID', 'FUA_NAME', 'cntAddresses', 'pc4',
       'INWONER', 'MAN', 'VROUW', 'INW_014', 'INW_1524', 'INW_2544',
       'INW_4564', 'INW_65PL', 'AANTAL_HH', 'GEM_HH_GR', 'NUTS2CODE',
       'NUTS2NAME', 'NUTS1CODE', 'NUTS1NAME', 'area_type'
    """
    data = pd.read_excel('Data/Neighbourhood_Descriptives.xlsx')
    return data

def UFS_Universe_NL():
    """
    Returns the UFS_Universe_nl.xlsx file as a pd dataframe.
    Note: I added the 'pc4' variable here by splitting postalCode

    Variables in order of columns:
    'operatorId', 'name', 'address', 'postalCode', 'city', 'Latitude',
       'Longitude', 'globalChannel', 'cuisineType', 'closed', 'pc4'
    """
    data = pd.read_excel("Data/UFS_Universe_NL.xlsx")

    # Adding pc4 variable
    data['pc4'] = data['postalCode'].str.extract(r'(\d{4})', expand=False) # using regex pattern to find the code
    return data

def UFS_Universe_NL_ratings():
    """
    Returns the UFS_Universe_nl.xlsx file as a pd dataframe.
    Variables in order of columns:
    'operatorId', 'name', 'address', 'postalCode', 'city', 'Latitude',
       'Longitude', 'globalChannel', 'cuisineType', 'closed', 'pc4', 'rating', 'no_reviews'
    """
    data = pd.read_csv("Data/ScrapedRatings/data_ratings.csv")

    return data

def zipcode_data_2017():
    """
    Returns the zipcode_data_2017.csv file as a pd dataframe.
    Note: I added the 'pc4' variable here by splitting postalCode

    Variables in order of columns:
        'pc4', 'INWONER', 'MAN', 'VROUW', 'INW_014', 'INW_1524', 'INW_2544',
       'INW_4564', 'INW_65PL', 'GEBOORTE', 'P_NL_ACHTG', 'P_WE_MIG_A',
       'P_NW_MIG_A', 'AANTAL_HH', 'TOTHH_EENP', 'TOTHH_MPZK', 'HH_EENOUD',
       'HH_TWEEOUD', 'GEM_HH_GR', 'WONING', 'WONVOOR45', 'WON_4564',
       'WON_6574', 'WON_7584', 'WON_8594', 'WON_9504', 'WON_0514', 'WON_1524',
       'WON_MRGEZ', 'P_HUURWON', 'P_KOOPWON', 'WON_HCORP', 'WON_NBEW',
       'WOZWONING', 'G_GAS_WON', 'G_ELEK_WON', 'UITKMINAOW', 'AFS_SUPERM',
       'AV1_SUPERM', 'AV3_SUPERM', 'AV5_SUPERM', 'AFS_DAGLMD', 'AV1_DAGLMD',
       'AV3_DAGLMD', 'AV5_DAGLMD', 'AFS_WARENH', 'AV5_WARENH', 'AV10WARENH',
       'AV20WARENH', 'AFS_CAFE', 'AV1_CAFE', 'AV3_CAFE', 'AV5_CAFE',
       'AFS_CAFTAR', 'AV1_CAFTAR', 'AV3_CAFTAR', 'AV5_CAFTAR', 'AFS_HOTEL',
       'AV5_HOTEL', 'AV10_HOTEL', 'AV20_HOTEL', 'AFS_RESTAU', 'AV1_RESTAU',
       'AV3_RESTAU', 'AV5_RESTAU', 'AFS_BSO', 'AV1_BSO', 'AV3_BSO', 'AV5_BSO',
       'AFS_KDV', 'AV1_KDV', 'AV3_KDV', 'AV5_KDV', 'AFS_BRANDW', 'AFS_OPRIT',
       'AFS_TRNOVS', 'AFS_TREINS', 'OAD', 'STED'
    """
    data = pd.read_csv('Data/zipcode_data_cbs/zipcode_data_2017.csv', sep='|')
    data = data.rename(columns={'PC4': 'pc4'}) # rename PC4 to pc4 to make it consistent with the other datasets
    return data

def zipcode_data_2019():
    """
    Returns the zipcode_data_2019.csv file as a pd dataframe.
    Note: I added the 'pc4' variable here by splitting postalCode

    Variables in order of columns:
        'pc4', 'INWONER', 'MAN', 'VROUW', 'INW_014', 'INW_1524', 'INW_2544',
       'INW_4564', 'INW_65PL', 'GEBOORTE', 'P_NL_ACHTG', 'P_WE_MIG_A',
       'P_NW_MIG_A', 'AANTAL_HH', 'TOTHH_EENP', 'TOTHH_MPZK', 'HH_EENOUD',
       'HH_TWEEOUD', 'GEM_HH_GR', 'WONING', 'WONVOOR45', 'WON_4564',
       'WON_6574', 'WON_7584', 'WON_8594', 'WON_9504', 'WON_0514', 'WON_1524',
       'WON_MRGEZ', 'UITKMINAOW', 'OAD', 'STED'
    """
    data = pd.read_csv('Data/zipcode_data_cbs/zipcode_data_2019.csv', sep='|')
    data = data.rename(columns={'PC4': 'pc4'}) # rename PC4 to pc4 to make it consistent with the other datasets
    return data

zipcode_data_2019()
zipcode_data_2017()
Neighborhood_Descriptives()
UFS_Universe_NL()
UFS_Universe_NL_ratings()