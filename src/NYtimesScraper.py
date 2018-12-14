'''
Scrape NY Times using NY Times API
'''
from nytimesarticle import articleAPI
import time
import csv
api = articleAPI('2d70d965acb44242ad6fc63d697d583a')

def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    try:
        for i in articles['response']['docs']:
            dic = {}
            dic['id'] = i['_id']
            #if i['abstract'] is not None:
            #    dic['abstract'] = i['abstract'].encode("utf8")
            dic['headline'] = i['headline']['main'].encode("utf8")
            #dic['desk'] = i['news_desk']
            dic['date'] = i['pub_date'][0:10] # cutting time of day.
            #dic['section'] = i['section_name']
            if i['snippet'] is not None:
                dic['snippet'] = i['snippet'].encode("utf8")
                dic['source'] = i['source']
                dic['type'] = i['type_of_material']
                dic['url'] = i['web_url']
                dic['word_count'] = i['word_count']
                # locations
                locations = []
                for x in range(0,len(i['keywords'])):
                    if 'glocations' in i['keywords'][x]['name']:
                        locations.append(i['keywords'][x]['value'])
                        dic['locations'] = locations
                        # subject
                        subjects = []
                        for x in range(0,len(i['keywords'])):
                            if 'subject' in i['keywords'][x]['name']:
                                subjects.append(i['keywords'][x]['value'])
                                dic['subjects'] = subjects
                                news.append(dic)
    except:
        pass
    return(news)

def get_articles(date,query):
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    all_articles = []

    for i in range(0,100): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        try:
            articles = api.search(q = query,
                                  fq = {'source':['Reuters','AP', 'The New York Times']},
                                  begin_date = date + '0101',
                                  end_date = date + '1231',
                                  sort='oldest',
                                  news_desk = 'business',
                                  subject = 'business',
                                  glocations = 'U.S.',
                                  page = str(i))
            articles = parse_articles(articles)
            time.sleep(1)
            all_articles = all_articles + articles
        except:
            pass
    return(all_articles)


def main():
    Ticker = []
    for i in range(2016,2018):
        try:
            print 'Processing' + str(i) + '...'
            Ticker_year =  get_articles(str(i),'stock market activity')
            Ticker = Ticker + Ticker_year
            keys = Ticker[0].keys()
            with open('Market1.csv', 'wb') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(Ticker)
        except :
            pass
if __name__ == '__main__':
    main()
