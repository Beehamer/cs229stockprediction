'''
Scrape Google News using bs4
'''
from requests import get
# from contextlib import closing
from bs4 import BeautifulSoup
import requests
from lxml import html
import pandas as pd
import time
import datetime as datetime
import random

timelist = pd.read_csv('techindicators_20.csv')['Date'][:1259]
Ticker = 'Paypal'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
to_merge = pd.DataFrame({'Date': [],'Ticker': [], 'Url':[], 'headline': [], 'source': [], 'snippet': []})
it = 0
for date in timelist:
    it += 1
    #if it%100 == 0:
    #    print('Fetching ',timelist[it])

    time.sleep(abs(random.gauss(0,2)))

    # Create URL
    Date = str(date).replace("-","")
    URL = 'https://www.google.com/search?q='+Ticker+'+'+Date+'&source=lnms&tbm=nws'

    # Get URL
    raw_html = requests.get(URL, headers=headers)
    soup = BeautifulSoup(raw_html.text, 'lxml')
    title_list = soup.find_all("a", class_="l lLrAF")
    date_list = soup.find_all("span", class_="f nsa fwzPFf")
    source_list = soup.find_all("span", class_="xQ82C e8fRJf")
    snippet_list = soup.find_all("div", class_="st")

    # Parse elements
    title = []
    date = []
    source = []
    snippet = []
    url = []
    for elem in title_list:
        valid = str(elem).replace("<em>", "").replace("</em>", "")[19:-4]
        out1 = valid.split("=")[1][1:-6]
        out2 = valid.split("=")[-1].split('>')[-1]
        url.append(out1)
        title.append(out2)
    for elem in date_list:
        valid = str(elem)[27:-7]
        date.append(valid)
    for elem in source_list:
        valid = str(elem)[27:-7]
        source.append(valid)
    for elem in snippet_list:
        valid = str(elem).replace("<em>", "").replace("</em>", "")[16:-10]
        snippet.append(valid)
    if len(title) == len(date) and len(title) == len(url) and len(title) == len(source) and len(title) == len(snippet):
        ticker = ['PYPL']*len(title)
        news = pd.DataFrame({'Date':date,'Ticker': ticker,'Url': url, 'headline':title, 'source': source, 'snippet': snippet})
        to_merge = pd.concat([to_merge,news])

to_merge.to_csv('PYPL_news.csv')
