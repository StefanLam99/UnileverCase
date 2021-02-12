import numpy as np
from DataSets import zipcode_data_2017, zipcode_data_2019, downloaded_zipcode_2017, medianincome_zipcode_data
import pandas as pd
import sys

zipcode_df = zipcode_data_2019()
cbs_2017 = zipcode_data_2017()
cbs_2017_extra = downloaded_zipcode_2017()
median = medianincome_zipcode_data()

print(zipcode_df)

# Drop all irrelevant variables
zipcode_df = zipcode_df.drop(columns=['AANTAL_HH', 'TOTHH_EENP', 'TOTHH_MPZK', 'HH_EENOUD', 'HH_TWEEOUD', 'WONING', 'WON_MRGEZ', 'GEBOORTE', 'STED', 'WONVOOR45', 'WON_4564', 'WON_6574', 'WON_7584', 'WON_8594', 'WON_9504', 'WON_0514', 'WON_1524'])

#Add extra income-related variables

zipcode_df['P_LINK_HH'] = cbs_2017_extra['P_LINK_HH']
zipcode_df['P_HINK_HH'] = cbs_2017_extra['P_HINK_HH']

# Add variables from 2017 CBS file (provided by UFS)
#zipcode_df['AFS_CAFE']= cbs_2017['AFS_CAFE']
#zipcode_df['AV1_CAFE']= cbs_2017['AV1_CAFE']
#zipcode_df['AV3_CAFE']= cbs_2017['AV3_CAFE']
zipcode_df['AV5_CAFE']= cbs_2017['AV5_CAFE']

#zipcode_df['AFS_CAFTAR']= cbs_2017['AFS_CAFTAR']
#zipcode_df['AV1_CAFTAR']= cbs_2017['AV1_CAFTAR']
#zipcode_df['AV3_CAFTAR']= cbs_2017['AV3_CAFTAR']
zipcode_df['AV5_CAFTAR']= cbs_2017['AV5_CAFTAR']

#zipcode_df['AFS_HOTEL']= cbs_2017['AFS_HOTEL']
zipcode_df['AV5_HOTEL']= cbs_2017['AV5_HOTEL']
#zipcode_df['AV10_HOTEL']= cbs_2017['AV10_HOTEL']
#zipcode_df['AV20_HOTEL']= cbs_2017['AV20_HOTEL']

#zipcode_df['AFS_RESTAU']= cbs_2017['AFS_RESTAU']
#zipcode_df['AV1_RESTAU']= cbs_2017['AV1_RESTAU']
#zipcode_df['AV3_RESTAU']= cbs_2017['AV3_RESTAU']
zipcode_df['AV5_RESTAU']= cbs_2017['AV5_RESTAU']

# Add the median income to the corresponding zipcode

set1 = list(median['pc4'])
set2 = list(zipcode_df['pc4'])
diff = list(set(set2)- set(set1))
zipcode_df["median_inc"] = np.nan
median.replace('.', np.nan, inplace=True)
j = 0
for i in range(median.shape[0]):
    if((zipcode_df.iloc[j]['pc4'] in diff) == False):

        zipcode_df.loc[zipcode_df.pc4 == set2[j], 'median_inc']  = median.iloc[i]['medianIncome']
        j = j+1
    else:
        #print(j)
        j = j + 1
        #print("added 1", j)

# Drop all rows with missing values (-99997) after turning the (-99997) into NaN
zipcode_df.replace(-99997, np.nan, inplace=True)
cbs_nan = zipcode_df
# amount of missing values per variable

print(cbs_nan.isnull().sum())
zipcode_df.dropna(inplace=True)

# Save as CSV

#zipcode_df.to_csv('zipcodedata_version_4_nanIncluded.csv', index = False, header = True)
# print(zipcode_df)

