import pandas as pd
import requests
from requests_html import HTMLSession
import re


def load_data(file_name):
    data = pd.read_excel(file_name, header =0)
    return data

def make_url(data:pd.DataFrame)->pd.DataFrame:
    """"
    This method takes the data framework and creates a google search url
    It returns a list with all the search URLS.
    """
    searchwords_list = data.loc[:, 'name'] + " "+ data.loc[:,'city']

    #Need regex to remove some non-alphabetic characters
    regex = re.compile('[^a-zA-Z ]')
    # First parameter is the replacement, second parameter is your input string

    base_url = "https://www.google.com/search?q="
    url_list = [base_url+regex.sub('', str(searchword)).replace(" ", "+") for searchword in searchwords_list]

    data['url_list'] = pd.DataFrame(url_list)
    return data

def get_rating_for_url(url:str) -> (str, str):
    """"
    For the given url string we start a HTML session. The page is rendered because there are
    Javascript elements on the page. Then we obtain the elements of interest
    """
    session = HTMLSession()
    r = session.get(url)
    r.html.render()
    try:
        rating = (r.html.find('.Aq14fc', first=True).text)
        no_reviews = (r.html.find('.hqzQac', first=True).text).split(" ")[0]
    except AttributeError:
        return (None,None)
    return (rating, no_reviews)

def get_all_ratings(data: pd.DataFrame):
    google_output = []
    # for url in data.loc[:, 'url_list']:
    for url in data.url_list:
        print(url)
        (rating, no_reviews) = get_rating_for_url(url)
        google_output.append([rating, no_reviews])
        print("{},{}", rating,no_reviews)
    return google_output


if __name__ == '__main__':
    file_name = r"C:\Users\bartd\Erasmus\Erasmus_\Jaar 4\Master Econometrie\Seminar\Code\Data\UFS_Universe_NL.xlsx"
    data = load_data(file_name)
    data = data.iloc[:5, :]
    data = make_url(data)
    # url_list = make_url(data)
    # test_url = url_list[1]
    # test_url = "https://www.google.com/search?q=Enfes+Amsterdam"
    google_output = get_all_ratings(data)
    review_dataframe = pd.DataFrame(google_output, columns=['rating', 'no_reviews'])
    data_new = pd.concat([data, review_dataframe], axis = 1)
    print(data_new)

# headers = requests.utils.default_headers()
# headers.update({ 'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
#                'Referer': "http://google.com"})