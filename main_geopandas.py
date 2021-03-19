import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from DataSets import UFS_Universe_NLnew
from Utils import rgb2hex
pd.set_option('display.expand_frame_repr', False)  # dataframes will be printed out without a line break

def main(version):
    """
    Main to obtain the map of the netherlands with the restaurants corresponding to the clusters of the best method
    """
    # initialization
    best_method = "SOM + k-means"  # our best method
    cities_nl = pd.read_csv('Data/nl.csv')  # data containing coordinates of the city in NL
    data_restaurants = UFS_Universe_NLnew()
    data_restaurants = data_restaurants[['pc4', 'Longitude', 'Latitude']]
    data_restaurants.dropna(inplace=True)
    pc4_labels = pd.read_csv('Results/pc4_best_labels_version_' + str(version)+ "_" + best_method + '.csv')
    pc4_labels["labels"] = pc4_labels["labels"] +1
    data_restaurants['pc4'] = data_restaurants['pc4'].astype(int)
    data_restaurants = data_restaurants.merge(pc4_labels, left_on='pc4', right_on='pc4')
    s = 1  # markersize
    clusters = [0, 1, 2 ,3, 5]


    # obtain the map of the netherlands
    zipfile = 'zip://Data/Provinciegrenzen_2019-shp.zip'
    municipal_boundaries = gpd.read_file(zipfile)
    p = municipal_boundaries.plot(color='bisque', edgecolor='white', legend=True)
    p.axis('off')
    gdf_cluster = gpd.GeoDataFrame(data_restaurants, geometry=gpd.points_from_xy(data_restaurants['Longitude'],data_restaurants['Latitude']))
    Netherlands = municipal_boundaries.geometry.unary_union
    #restaurants_within_Netherlands = gdf_cluster.within(Netherlands)

    #restaurants_within_Netherlands.to_csv('Results/restaurants_within_netherlands.csv', header=False)
    restaurants_within_Netherlands = pd.read_csv('Results/restaurants_within_netherlands.csv', header=None).iloc[:,1]


    within_Netherlands = gdf_cluster.loc[restaurants_within_Netherlands]
    labels= pc4_labels["labels"].values
    #  gdf_cluster.plot(ax=p, column='labels', categorical=True, legend=True, legend_kwds={'title': "Cluster"}, markersize=s, vmin=0, vmax=6, cmap= 'Paired')  # vmin and vmax colormap can do with cate
    within_Netherlands.plot(ax=p, column="labels", categorical = True, legend=True, legend_kwds={'title': "Cluster"}, markersize=s, vmin=-1, vmax=5, cmap= 'Paired')  # vmin and vmax colormap can do with cate
    for i in range (len(cities_nl)):
        city = cities_nl.iloc[i,:]
        if (city["capital"] != "minor" and city["lng"]>0 and city['city'] != 'Haarlem') or city['city']== 'Rotterdam' :
            plt.plot([city['lng']],[city['lat']], 'o', color = 'black', markersize=3)
            plt.annotate(" " + str(city['city']), xy= (city['lng'], city['lat'] ), fontsize=8)
    plt.savefig("Results/Clusters_on_Netherlands.png")
    plt.show()


    #  plot without the restaurants
    municipal_boundaries = gpd.read_file(zipfile)
    p = municipal_boundaries.plot( edgecolor='white', legend=True)
    p.axis('off')
    for i in range (len(cities_nl)):
        city = cities_nl.iloc[i,:]
        if (city["capital"] != "minor" and city["lng"]>0 and city['city'] != 'Haarlem') or city['city']== 'Rotterdam' :
            plt.plot([city['lng']],[city['lat']], 'o', color = 'black')
            plt.annotate(" " + str(city['city']), xy= (city['lng'], city['lat'] ))
    plt.show()
if __name__ == '__main__':
    version = 10
    main(version)