import pandas as pd
from requests_html import HTMLSession
import re
import multiprocessing as mp
from datetime import datetime, date
import time

class RatingScraper:
    data = pd.DataFrame()
    data_ratings = pd.DataFrame()
    google_output_df = pd.DataFrame()

    file_output = str

    def __init__(self, file_input, file_output):
        self.data = self.load_data(file_input)
        self.file_output = file_output
        self.start = time.time()

    def load_data(self, file_name):
        data = pd.read_excel(file_name, header =0)
        return data

    def write_data(self):
        self.data_ratings.to_csv(self.file_output)

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
        url = url[0]
        session = HTMLSession()
        r = session.get(url)
        r.html.render()
        try:
            rating = (r.html.find('.Aq14fc', first=True).text)
            no_reviews = (r.html.find('.hqzQac', first=True).text).split(" ")[0]
        except AttributeError:
            return (None,None)
        return [rating, no_reviews]


    def get_all_ratings(self):
        """"
        Go through the urls in data.url_list, for every url we call get_rating_for_url and get the
        ratings for that url. It creates the instance variable google_output_df, an dataframe with the ratings
        for every restaurant.
        """
        google_output = []
        # for url in data.loc[:, 'url_list']:
        for url in self.data.url_list:
            print(url)
            (rating, no_reviews) = self.get_rating_for_url(url)
            google_output.append([rating, no_reviews])
            print(rating, no_reviews)
        google_output = pd.DataFrame(google_output, columns=['rating', 'no_reviews'])
        self.google_output_df = google_output

    def get_all_ratings_mp(self):
        """"
        Multiprocessing code. Maximum number of processors is used to get all the url data
        I did not implement test to check if it returns the review in the same order as
        with a simple for loop (#TODO?)
        """
        pool = mp.Pool(mp.cpu_count())
        google_output = pool.map(RatingScraper.get_rating_for_url,[(url,) for url in self.data.url_list])
        self.google_output_df = pd.DataFrame(google_output, columns=['rating', 'no_reviews'])

    def scrape(self):
        """"
        This method is the 'main method' of the class. It calls all the class methods to obtain the final dataframe
        this dataframe then is written to a csv file.
        """
        self.make_url()
        self.get_all_ratings_mp()
        self.data_ratings = pd.concat([self.data, self.google_output_df], axis = 1)
        self.write_data()
        print("Succesfully written data. The program run for ", round(time.time() - self.start, 3), " seconds")


if __name__ == '__main__':
    file_input = r"C:\Users\bartd\Erasmus\Erasmus_\Jaar 4\Master Econometrie\Seminar\Code\Data\UFS_Universe_NL.xlsx"
    file_output = r"C:\Users\bartd\Erasmus\Erasmus_\Jaar 4\Master Econometrie\Seminar\Code\Data\data_ratings.csv"
    scraper = RatingScraper(file_input, file_output)
    scraper.scrape()
