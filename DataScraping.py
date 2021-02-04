import pandas as pd
from requests_html import HTMLSession
from DataSets import UFS_Universe_NL, zipcode_data_2017, zipcode_data_2019, Neighborhood_Descriptives
import re
import multiprocessing as mp
from datetime import datetime, date
import time


# Note: Make sure to install: pip install -U "urllib3<1.25", otherwise you might get an error.

class RatingScraper:
    data = pd.DataFrame()
    data_ratings = pd.DataFrame()
    google_output_df = pd.DataFrame()

    file_output = str

    def __init__(self, file_output = 'data_ratings'):
        self.data = UFS_Universe_NL()
        self.file_output = 'Data/ScrapedRatings/' + file_output + '.csv'
        self.start = time.time()

    def make_url(self):
        """"
        This method takes the data framework and creates a google search url
        It returns a list with all the search URLS.
        """
        searchwords_list = self.data.loc[:, 'name'] + " "+ self.data.loc[:,'city']

        #Need regex to remove some non-alphabetic characters
        regex = re.compile('[^a-zA-Z ]')
        # First parameter is the replacement, second parameter is your input string

        base_url = "https://www.google.com/search?q="
        url_list = [base_url+regex.sub('', str(searchword)).replace(" ", "+") for searchword in searchwords_list]

        self.data['url_list'] = pd.DataFrame(url_list)

    @staticmethod
    def get_rating_for_url(url:list) -> list:
        """"
        For the given url string we start a HTML session. The page is rendered because there are
        Javascript elements on the page. Then we obtain the elements of interest
        """
        #url = url[0]
        session = HTMLSession()
        r = session.get(url)
        r.html.render()
        try:
            rating = (r.html.find('.Aq14fc', first=True).text)
            no_reviews = (r.html.find('.hqzQac', first=True).text).split(" ")[0]
        except AttributeError:
            return (None,None)
        return [rating, no_reviews]


    def get_all_ratings(self, begin, end):
        """"
        Go through the urls in data.url_list, for every url we call get_rating_for_url and get the
        ratings for that url. It creates the instance variable google_output_df, an dataframe with the ratings
        for every restaurant.
        """
        google_output = []
        # for url in data.loc[:, 'url_list']:
        for url in self.data.url_list[begin:end]:
            print(url)
            print(type(url))
            (rating, no_reviews) = self.get_rating_for_url(url)
            google_output.append([rating, no_reviews])
            print(rating, no_reviews)
        google_output = pd.DataFrame(google_output, columns=['rating', 'no_reviews'])
        self.google_output_df = google_output

    def get_all_ratings_mp(self, begin, end):
        """"
        Multiprocessing code. Maximum number of processors is used to get all the url data
        I did not implement test to check if it returns the review in the same order as
        with a simple for loop (#TODO?)
        """
        pool = mp.Pool(mp.cpu_count())
        google_output = pool.map(RatingScraper.get_rating_for_url,[url for url in self.data.url_list[begin: end]])
        self.google_output_df = pd.DataFrame(google_output, columns=['rating', 'no_reviews'])

    def scrape(self, begin = 0, end = 29055, multiprocessing = True): # 29055 is the number of restaurants we have
        """"
        This method is the 'main method' of the class. It calls all the class methods to obtain the final dataframe
        this dataframe then is written to a csv file.
        """
        self.make_url()
        if(multiprocessing):
            self.get_all_ratings_mp(begin, end)
        else:
            self.get_all_ratings(begin, end)

        self.data_ratings = pd.concat([self.data, self.google_output_df], axis = 1)
        self.data_ratings.to_csv(self.file_output) # create csv file with the ratings and number of reviews
        print("Succesfully written data. The program run for ", round(time.time() - self.start, 3), " seconds")


if __name__ == '__main__':
    file_output = 'data_ratings' # give ur output csv. file a name
    scraper = RatingScraper(file_output)
    scraper.scrape(begin=0, end=5000, multiprocessing=False)
