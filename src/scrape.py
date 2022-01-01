from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
from datetime import date
import pickle
import os

'''
This Python script scrapes news articles from the front pages and RSS feeds of the following publishers:

New York Post
CNN
Fox News
One America News Network
Associated Press
Palmer Report
The Daily Beast
Occupy Democrats
Epoch Times
'''

def epoch_times_frontpage_scrape(url):
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'lxml')

    urls = []

    for div1 in soup.find_all('div', attrs={"class": "article_info text"}):
        for div2 in div1.find_all('div', attrs={"class": "title"}):
            for link in div2.find_all('a'):
                urls.append(link['href'])

    return urls


def epoch_times_scrape(url):
    obj = requests.get(url).text

    soup = BeautifulSoup(obj, 'lxml')
    text = ""
    headline = ""

    # find headline
    for div in soup.find_all('div', attrs={"class": "post_title"}):
        for h1 in div.find_all('h1'):
            headline = h1.text
            headline = headline.strip()

    for div in soup.find_all('div', attrs={"class": "post_content"}):
        for para in div.find_all('p'):
            paragraph_text = para.text
            text = text + paragraph_text + " "

    return headline, text

def CNN_scrapeRSS(url, numarticles=0):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'xml')

    # find all news items in RSS feed
    news_list = soup.findAll("item")

    # lists for storage
    links = []
    urls = []

    # get links to articles from RSS feed
    for each in news_list:
        if "index.html" in each.link.text:
            links.append(each.link.text)

    # if numarticles = 0, return all links found in feed
    if numarticles == 0:
        for i in range(len(links)):
            urls.append(links[i])
    # else, return numarticles links found in feed
    else:
        for i in range(0, numarticles):
            urls.append(links[i])

    return urls

def cnn_scrape(url):
    # get source
    obj = requests.get(url).text

    # CNN's news articles are located within the div classes 'zn-body...' and there exist
    # two different ones, so all the div classes for which news articles exist in are
    # replaced with 'scrapeme' in the source code
    obj = obj.replace('zn-body__paragraph speakable', 'scrapeme')
    obj = obj.replace('zn-body__paragraph', 'scrapeme')
    obj = obj.replace('—', '-')

    soup = BeautifulSoup(obj, 'lxml')

    # find headline
    for row in soup.find_all('h1', attrs={"class": "pg-headline"}):
        headline = row.text

    # find all news divs
    for row in soup.find_all('p', attrs={"class": "scrapeme"}):
        text = row.text + " "

    # find all news paragraphs
    for row in soup.find_all('div', attrs={"class": "scrapeme"}):
        text = text + row.text + " "

    # CNN's article formatting is rather unreliable, so sometimes nothing is scraped... :(
    # try/except - if the replace statement throws an error (i.e., text is empty), then write error
    try:
        text = text.replace("  ", " ")
        return headline, text
    except:
        return "error", "error"


def FOX_scrapeRSS(url, numarticles=0):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'xml')

    # get all links in RSS feed
    news_list = soup.findAll("item")

    # storage lists
    links = []
    urls = []

    # get all links from source
    for each in news_list:
        links.append(each.link.text)

    # if numarticles = 0, return all links found in feed
    if numarticles == 0:
        for i in range(len(links)):
            urls.append(links[i])
    # else, return numarticles links found in feed
    else:
        for i in range(0, numarticles):
            urls.append(links[i])

    return urls


def fox_scrape(url):
    # get source
    obj = requests.get(url).text
    obj = obj.replace('—', '-')
    soup = BeautifulSoup(obj, 'lxml')
    text = ""

    # find headline
    for h1 in soup.find_all('h1', attrs={"class": "headline"}):
        headline = h1.text

    # get article content
    # find appropriate div with article content
    for divs in soup.find_all('div', attrs={"class": "article-body"}):
        # get paragraphs in div (actual article content)
        for para in divs.find_all('p'):
            paragraph_text = para.text
            # some "None" strings were being acquired, let's skip those...
            if paragraph_text == "None":
                continue
            else:
                text = text + paragraph_text + " "

    # replace unwated
    text = text.replace(u'\xa0', u' ')
    text = text.replace("Fox News Flash top headlines are here.", "")
    text = text.replace("Check out what's clicking on Foxnews.com.", "")

    return headline, text


def NYP_scrapeRSS(url, numarticles=0):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'xml')

    # get links in RSS feed
    news_list = soup.findAll("link")

    # storage list
    urls = []

    # iterate over links, do not include just link to nypost.com
    for each in news_list:
        if each == "https://nypost.com":
            continue
        else:
            urls.append(each.text)

    # filter out those scraped as NoneType
    urls = list(filter(None, urls))

    # nypost.com was still being acquired as the first link, return only index 1 onwards
    return urls[1:]


def nyp_scrape(url):
    # get source
    obj = requests.get(url).text
    # replace fancy hyphen with regular hyphen
    obj = obj.replace('—', '-')

    # create soup object of source
    soup = BeautifulSoup(obj, 'lxml')
    text = ""

    # find headline
    for h1 in soup.find_all('h1', attrs={"class": "headline headline--single"}):
        headline = h1.text

    # get article content
    # find appropriate div with article content
    for divs in soup.find_all('div', attrs={"class": "single__content entry-content m-bottom"}):
        # get paragraphs in div (actual article content)
        for para in divs.find_all('p'):
            paragraph_text = para.text
            # some "None" strings were being acquired, let's skip those...
            if paragraph_text == "None":
                continue
            else:
                text = text + paragraph_text + " "

    # replace unwanted
    text = text.replace(u'\xa0', u' ')
    text = text.replace("With Post wires", "")
    headline = headline.replace("\n", "")
    headline = headline.replace("\t", "")

    return headline, text


def OAN_scrapeRSS(url):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'xml')

    # get links from RSS feed
    news_list = soup.findAll("link")
    urls = []

    # get each link, skip only oann.com
    for each in news_list:
        if each == "https://www.oann.com":
            continue
        else:
            urls.append(each.text)

    # filter out any acquired with NoneType
    urls = list(filter(None, urls))

    # oann.com was still being returned as the first two links, return all after those two
    return urls[2:]


def oan_scrape(url):
    # get source
    obj = requests.get(url).text

    # replace fancy hyphen with regular
    obj = obj.replace('—', '-')

    # create soup object
    soup = BeautifulSoup(obj, 'lxml')
    text = ""

    # find headline
    for h1 in soup.find_all('h1', attrs={"class": "entry-title"}):
        headline = h1.text

    # get divs with article content
    for divs in soup.find_all('div', attrs={"class": "entry-content clearfix"}):
        # get actual article content
        for para in divs.find_all('p'):
            paragraph_text = para.text
            if paragraph_text == "None":
                continue
            else:
                text = text + paragraph_text + " "

    # replace unwanted
    text = text.replace(u'\xa0', u' ')

    return headline, text


def daily_beast_frontpage_scrape(url):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'lxml')

    # storage lists
    urls = []

    # get div with links
    for divs in soup.find_all('div', attrs={"class": "GridStory__title-link"}):
        # get links to news articles
        for anchor in divs.find_all('a'):
            urls.append(anchor['href'])

    return urls


def fox_politics_scrape(url):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'lxml')

    # storage list
    urls = []

    # get article header/title
    for h4 in soup.find_all('h4', attrs={"class": "title"}):
        # find links in page, store in list
        for links in h4.find_all('a'):
            if "politics" in links['href']:
                if "video" not in links['href']:
                    urls.append("{0}{1}".format("https://www.foxnews.com", links['href']))

    return urls


def palmer_report_frontpage_scrape(url):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'lxml')

    # storage list
    urls = []

    # find posts on front page, store in list
    for h2 in soup.find_all('h2', attrs={"class": "fl-post-grid-title"}):
        for links in h2.find_all('a'):
            urls.append(links['href'])

    return urls


def palmer_report_scrape(url):
    # get source
    obj = requests.get(url).text
    obj = obj.replace('—', '-')
    soup = BeautifulSoup(obj, 'lxml')
    text = ""

    # find headline
    for h1 in soup.find_all('h1', attrs={"class": "fl-post-title"}):
        headline = h1.text
        headline = headline.strip()

    # get div with article content
    for divs in soup.find_all('div', attrs={"class": "fl-post-content clearfix"}):
        # get article text
        for para in divs.find_all('p'):
            paragraph_text = para.text
            if paragraph_text == "None":
                continue
            else:
                text = text + paragraph_text + " "

    # replace unwanted
    text = text.replace(u'\xa0', u' ')
    text = text.replace("Bill Palmer is the publisher of the political news outlet Palmer Report", "")

    return headline, text


def OD_frontpage_scrape(url):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'lxml')

    # storage list
    urls = []

    # get news article links from frontpage divs
    for divs in soup.find_all('div', attrs={"class": "post-title"}):
        # iterate over headers and scrape links, store
        for h in divs.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
            for links in h.find_all('a'):
                urls.append(links['href'])

    return urls


def OD_scrape(url):
    # get source
    obj = requests.get(url).text
    obj = obj.replace('—', '-')
    soup = BeautifulSoup(obj, 'lxml')

    # initialize variables
    text = ""
    headline = ""

    # find headline
    for h1 in soup.find_all('h1', attrs={"class": "entry-title"}):
        headline = h1.text

    # find div with article content
    for divs in soup.find_all('div', attrs={"class": "post-content-container"}):
        # scrape all paragraphs
        for para in divs.find_all('p'):
            paragraph_text = para.text
            # keep going if text == None and don't add to article body
            if paragraph_text == "None":
                continue
            else:
                text = text + paragraph_text + " "

    # replace unwanted
    text = text.replace(u'\xa0', u' ')
    text = text.replace("Sponsored Links ", "")
    text = text.replace("\n ", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    headline = headline.replace("\n", "")
    headline = headline.replace("\t", "")

    return headline, text

def daily_beast_scrape(url):
    # get source
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'lxml')

    # initialize variables
    text = ""
    headline = ""

    # find headline
    for h1 in soup.find_all('h1', attrs={"class": "StandardHeader__title"}):
        headline = h1.text
        headline = headline.strip()

    # handle 'premium' news articles
    if headline == "":
        for h1 in soup.find_all('h1', attrs={"class": "FeatureHeader__title"}):
            headline = h1.text
            headline = headline.strip()

    for divs in soup.find_all('div', attrs={"class": "Mobiledoc"}):
        for para in divs.find_all('p'):
            paragraph_text = para.text
            if paragraph_text == "None":
                continue
            else:
                text = text + paragraph_text + " "

    # replace unwated
    text = text.replace(u'\xa0', u' ')

    return headline, text


def ap_politics_page(url):

    # get source
    obj = requests.get(url).text

    # replace fancy punctuation with regular characters
    obj = obj.replace('”', '"')
    obj = obj.replace('“', '"')
    obj = obj.replace('’', "'")
    obj = obj.replace('—', '--')

    # create soup object
    soup = BeautifulSoup(obj, 'lxml')

    urls = []

    # find all links in the soup object, and if they are a news article (contain /article/ in the link),
    # then append them to the news list
    for links in soup.find_all('a', href=True):
        if "/article/" in links['href']:
            url = "http://apnews.com" + links['href']
            urls.append(url)

    # return urls and drop duplicates in process
    return list(set(urls))

# This function scrapes text from a news article from AP
def ap_scrape(url):

    # get source
    # adding 3 sec sleep to avoid "Max retries exceeded" error
    try:
        obj = requests.get(url).text

        # replace fancy punctuation with regular characters
        obj = obj.replace('”', '"')
        obj = obj.replace('“', '"')
        obj = obj.replace('’', "'")
        obj = obj.replace('—', '--')

        # create soup object
        soup = BeautifulSoup(obj, 'lxml')
        text = ""

        # get headline of news article
        for title in soup.find_all('title'):
            headline = title.text

        # iterate through paragraphs of news article and append to 'text' variable
        for divs in soup.find_all('div', attrs={"class": "Article"}):
            for para in divs.find_all('p'):
                text = text + para.text + " "

        # replace unwanted
        text = text.replace("  ", " ")
        text = text.replace("(AP)","")

        return headline, text
    except:
        print("***Connection Error")
        return "Error", "Error"



def adfontes_scrape(url):
    obj = requests.get(url).text
    soup = BeautifulSoup(obj, 'xml')

    reliabilitys = []
    biases = []

    for para in soup.find_all("p"):
        for strong in para.find_all("strong"):
            if "Reliability:" in strong.text:
                txt = strong.text
                txt = txt.split()
                rel = float(txt[1])
                reliabilitys.append(rel)
            if "Bias:" in strong.text:
                txt = strong.text
                txt = txt.split()
                bias = float(txt[1])
                biases.append(bias)

    return [reliabilitys[0], biases[0]]

def adfontes_dictionary(verbose=False):
    adfontes_urls = ["https://adfontesmedia.com/cnncom-bias-and-reliability/",
                     "https://adfontesmedia.com/fox-news-com-bias-and-reliability/",
                     "https://adfontesmedia.com/new-york-post-bias-and-reliability/",
                     "https://adfontesmedia.com/oann-one-america-news-network-bias-and-reliability/",
                     "https://adfontesmedia.com/occupy-democrats-bias-and-reliability/",
                     "https://adfontesmedia.com/palmer-report-bias-and-reliability/",
                     "https://adfontesmedia.com/daily-beast-bias-reliability/",
                     "https://adfontesmedia.com/ap-bias-and-reliability/",
                     "https://adfontesmedia.com/epoch-times-bias-and-reliability/"]

    adfontes_sources = ["CNN", "Fox News", "NY Post", "OAN", "Occupy Democrats", "Palmer Report", "The Daily Beast",
                        "Associated Press", "Epoch Times"]

    adfontes_bias = {}
    adfontes_reliability = {}

    if verbose:
        print("Scraping bias/reliability scores...")

    for i in range(len(adfontes_urls)):
        source = adfontes_sources[i]
        scores = adfontes_scrape(adfontes_urls[i])

        adfontes_reliability[source] = scores[0]
        adfontes_bias[source] = scores[1]

    if verbose:
        print("Pickling dictionary...")

    with open('bias.pickle', 'wb') as file:
        pickle.dump(adfontes_bias, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('reliability.pickle', 'wb') as file:
        pickle.dump(adfontes_reliability, file, protocol=pickle.HIGHEST_PROTOCOL)


def init_storage_date():
    headlines = []
    article = []
    source = []
    dates = []
    link_list = []

    today = date.today()
    the_date = today.strftime("%m/%d/%y")

    return headlines, article, source, dates, link_list, the_date

def rss(verbose=False):
    feeds = ["http://rss.cnn.com/rss/cnn_allpolitics.rss",
             "http://feeds.foxnews.com/foxnews/politics",
             "https://nypost.com/news/feed/",
             "https://www.oann.com/category/world/feed/"]

    headlines, article, source, dates, link_list, the_date = init_storage_date()

    today = date.today()
    the_date = today.strftime("%m/%d/%y")

    if verbose:
        print("Scraping CNN RSS Feed...")
    urls = CNN_scrapeRSS("http://rss.cnn.com/rss/cnn_allpolitics.rss")
    for url in urls:
        scrape = cnn_scrape(url)
        headlines.append(scrape[0])
        article.append(scrape[1])
        source.append("CNN")
        dates.append(str(the_date))
        link_list.append(url)

    if verbose:
        print("Scraping FOX RSS Feed...")
    urls = FOX_scrapeRSS("http://feeds.foxnews.com/foxnews/politics")
    for url in urls:
        scrape = fox_scrape(url)
        headlines.append(scrape[0])
        article.append(scrape[1])
        source.append("Fox News")
        dates.append(str(the_date))
        link_list.append(url)

    if verbose:
        print("Scraping NY Post RSS Feed...")
    urls = NYP_scrapeRSS("https://nypost.com/news/feed/")
    for url in urls:
        scrape = nyp_scrape(url)
        headlines.append(scrape[0])
        article.append(scrape[1])
        source.append("NY Post")
        dates.append(str(the_date))
        link_list.append(url)

    if verbose:
        print("Scraping OAN RSS Feed...")
    urls = OAN_scrapeRSS("https://www.oann.com/category/world/feed/")
    for url in urls:
        scrape = oan_scrape(url)
        headlines.append(scrape[0])
        article.append(scrape[1])
        source.append("OAN")
        dates.append(str(the_date))
        link_list.append(url)

    df = pd.DataFrame({"Source": source,
                       "Headline": headlines,
                       "Content": article,
                       "Date": dates,
                       "URL" : link_list})

    return df

def scrape_news(url,scraper,verbose=False):
    headlines, article, source, dates, link_list, the_date = init_storage_date()

    if scraper == "FOX":
        publisher = "Fox News"
        urls = fox_politics_scrape(url)
    elif scraper == "PR":
        publisher = "Palmer Report"
        urls = palmer_report_frontpage_scrape(url)
    elif scraper == "OD":
        publisher = "Occupy Democrats"
        urls = OD_frontpage_scrape(url)
    elif scraper == "AP":
        publisher = "Associated Press"
        urls = ap_politics_page(url)
    elif scraper == "DB":
        publisher = "The Daily Beast"
        urls = daily_beast_frontpage_scrape(url)
    elif scraper == "ET":
        publisher = "Epoch Times"
        urls = epoch_times_frontpage_scrape(url)

    if verbose:
        print("Fetching {0} news from page...".format(publisher))

    for link in urls:

        if scraper == "FOX":
            scrape = fox_scrape(link)
        elif scraper == "PR":
            scrape = palmer_report_scrape(link)
        elif scraper == "OD":
            scrape = OD_scrape(link)
        elif scraper == "AP":
            scrape = ap_scrape(link)
        elif scraper == "DB":
            scrape = daily_beast_scrape(link)
        elif scraper == "ET":
            scrape = epoch_times_scrape(link)

        headlines.append(scrape[0])
        article.append(scrape[1])
        source.append(publisher)
        dates.append(str(the_date))
        link_list.append(link)

    df = pd.DataFrame({"Source": source,
                       "Headline": headlines,
                       "Content": article,
                       "Date": dates,
                       "URL" : link_list})

    return df

def gather_csv(middle=False):
    path = "/Users/nawal/PycharmProjects/NawalPython/605_744_Information_Retrieval/Project/archive"
    text_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    cat = []

    for file in text_files:
        frame = pd.read_csv(file)
        cat.append(frame)

    dframe = pd.concat(cat)

    dframe = dframe.astype({"Headline": "str",
                            "Content": "str"})

    dframe.reset_index(drop=True, inplace=True)
    dframe = dframe.drop_duplicates()
    dframe = dframe.drop_duplicates(subset='Headline', keep="last")
    dframe = dframe.drop_duplicates(subset='Content', keep="last")
    dframe.reset_index(drop=True, inplace=True)

    dframe = dframe[dframe["Headline"] != "Error"]
    dframe.reset_index(drop=True, inplace=True)

    with open('bias.pickle', 'rb') as file:
        adfontes_bias = pickle.load(file)

    with open('reliability.pickle', 'rb') as file:
        adfontes_reliability = pickle.load(file)

    pubs = dframe["Source"].to_list()

    if not middle:
        leaning = []
        leaning_encode = []
        for each in pubs:
            rating = adfontes_bias[each]
            if rating > 0:
                leaning.append("Right")
                leaning_encode.append(1)
            else:
                leaning.append("Left")
                leaning_encode.append(0)

    if middle:
        leaning = []
        leaning_encode = []
        for each in pubs:
            rating = adfontes_bias[each]
            if rating > 6:
                leaning.append("Right")
                leaning_encode.append(1)
            elif rating < -6:
                leaning.append("Left")
                leaning_encode.append(0)
            else:
                leaning.append("Middle")
                leaning_encode.append(2)

    dframe["Leaning"] = pd.Series(leaning)
    dframe["Leaning Encode"] = pd.Series(leaning_encode)
    dframe = dframe.drop(["URL"],axis=1)

    for i in range(dframe.shape[0]):
        dframe["Headline"].iloc[i].replace(" | AP News", "")
        dframe["Headline"].iloc[i].replace("Associated Press", "")
        dframe["Headline"].iloc[i].replace("Palmer Report", "")
        dframe["Headline"].iloc[i].replace("Occupy Democrats", "")
        dframe["Headline"].iloc[i].replace("Fox News", "")
        dframe["Headline"].iloc[i].replace("CNN", "")

    return dframe, adfontes_bias, adfontes_reliability

def main(verbose=False):
    if verbose:
        print("Running master scraper...")

    dfs = []

    rss_news = rss(verbose=verbose)
    fox_news = scrape_news(url="https://www.foxnews.com/politics", scraper="FOX", verbose=verbose)
    palmer_news = scrape_news(url="https://www.palmerreport.com/", scraper="PR", verbose=verbose)
    OD_news = scrape_news(url="https://occupydemocrats.com/", scraper="OD", verbose=verbose)
    beast_news = scrape_news(url="https://www.thedailybeast.com/category/politics", scraper="DB", verbose=verbose)
    press_news = scrape_news(url="https://apnews.com/hub/politics?utm_source=apnewsnav&utm_medium=navigation", scraper="AP", verbose=verbose)
    epoch_news1 = scrape_news(url="https://www.theepochtimes.com/c-policy?utm_source=hot_topics_rec&utm_medium=frnt_top", scraper="ET", verbose=verbose)
    epoch_news2 = scrape_news(url="https://www.theepochtimes.com/c-policy2?utm_source=hot_topics_rec&utm_medium=frnt_top", scraper="ET", verbose=verbose)
    epoch_news3 = scrape_news(url="https://www.theepochtimes.com/c-policy3?utm_source=hot_topics_rec&utm_medium=frnt_top", scraper="ET", verbose=verbose)
    epoch_news4 = scrape_news(url="https://www.theepochtimes.com/c-policy4?utm_source=hot_topics_rec&utm_medium=frnt_top", scraper="ET", verbose=verbose)

    dfs.append(fox_news)
    dfs.append(palmer_news)
    dfs.append(OD_news)
    dfs.append(beast_news)
    dfs.append(rss_news)
    dfs.append(press_news)
    dfs.append(epoch_news1)
    dfs.append(epoch_news2)
    dfs.append(epoch_news3)
    dfs.append(epoch_news4)

    all_news = pd.concat(dfs)

    all_news.reset_index(drop=True, inplace=True)
    all_news.drop("URL", 1, inplace=True)
    now = datetime.now()
    scrape_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = "master_{0}.csv".format(str(scrape_time))
    all_news.to_csv(name, index=None)

    if verbose:
        print("")
        print("# of news articles scraped: ", all_news.shape[0])
        print("")
        print("Breakdown of articles scraped by source:")
        print(all_news["Source"].value_counts())
        print("")

main(verbose=True)
adfontes_dictionary(verbose=True)

print("")
print("Scraping complete!")

dframe, bias, reliability = gather_csv()
print("")
print("Total collection size: ", dframe.shape[0])
