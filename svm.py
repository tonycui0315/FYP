# Libraries

# Web scraper libraries
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import sys
import csv

# ML libraries
import re 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC
import matplotlib.pyplot as plt

# NLTK Stemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Logistic Regression
lreg = LogisticRegression()



def get_page_links(pageDriver, baseUrl):
    a_tags = driver.find_elements_by_tag_name('a')

    # Get all urls
    urls = [tag.get_attribute('href') for tag in a_tags]

    #Get urls that start with baseUrl
    useful_urls = [url for url in urls if url and baseUrl in url]

    return useful_urls

def get_page_text(pageDriver):
    pageBody = pageDriver.find_element_by_xpath("/html/body")
    if not pageBody:
        return ""
    return pageBody.text


visited_urls = set()
unvisited_urls = set()
error_urls = set()

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument('--log-level=1')
driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)

baseUrl = input("Please enter url: ")
unvisited_urls.add(baseUrl)

ct = 0
while unvisited_urls:
    # Remove url from unvisited and add it to visited
    currPage = unvisited_urls.pop()
    print(currPage)
    visited_urls.add(currPage)

    # Get page content
    try:
        driver.get(currPage)
    except:
        error_urls.add(currPage)
        print('Error getting page: ', currPage)
        continue

    # Get links
    try:
        currLinks = get_page_links(driver, baseUrl)
    except:
        error_urls.add(currPage)
        print('Error getting links: ', currPage)
        continue

    # Get text
    try:
        currText = get_page_text(driver)
        textToWrite = currText.splitlines()
        with open(f'testCSV.csv', 'w') as csvfile:
            fieldnames = ['id', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for text in textToWrite:
                writer.writerow({'id': ct, 'text': {text}})
                ct += 1
    except:
        error_urls.add(currPage)
        print('Error getting text: ', currPage)
        continue

print("WEB SCRAPING FINISHED")

# Read data pandas, keep raw copy of data
train_RAW = pd.read_csv('train.csv')
test_RAW = pd.read_csv('testCSV.csv')
train = train_RAW
test = test_RAW

# TF-IDF freq
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.90, 
    min_df=2, 
    max_features=6000, 
    stop_words='english'
    )
# Bag Of Words Model
bow_vectorizer = CountVectorizer(
    max_df=0.90, 
    min_df=2, 
    max_features=6000, 
    stop_words='english'
    )

# Combine training data with test data
allTexts = train.append(test, ignore_index=True, sort=False)

def fnc_clean_tweet(tweet_txt, pattern):
    r = re.findall(pattern, tweet_txt)
    for i in r:
        tweet_txt = re.sub(i, '', tweet_txt)  # Substitute pattern with blank

    return tweet_txt

# Clean text 
# Add new column 'tidy_tweet'

# Clean user handles (@user_name)
allTweets['tidy_text'] = np.vectorize(fnc_clean_tweet)(
    allTweets['text'], "@[\w]*") 

# Clean non-alphabets
allTweets['tidy_text'] = allTweets['tidy_text'].str.replace(
    "[^a-zA-Z]", " ") 


# Tokenization
tokenized_text = allTweets['tidy_text'].apply(lambda x: x.split())

tokenized_text = tokenized_text.apply(
    lambda x: [lemmatizer.lemmatize(i) for i in x])  

# Combine tokens back together
for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])

# Replace with lemmatized
allTweets['tidy_text'] = tokenized_text


# Feature extraction - Bag-of-Words [sklearn’s CountVectorizer]
# ------------------------------------------------------------------------------------------------------------------

# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(allTweets['tidy_text'])


# TF-IDF Features - Looks at frequency of occurence for terms/importance of the term

# TF = (Number of times term t appears in a document)/(Number of terms in the document)
# IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
# 
# TF-IDF = TF*IDF

# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(allTweets['tidy_text'])

train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

# splitting data into training and validation set



# xtrain_bow training dataset
# xvalid_bow validation for training
# ytrain label vector
# yvalid validation label vector
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(
    train_bow, train['label'], test_size=0.1)

# train_bow feature matrix
# train['label'] label vector
# random_state shuffles data before split into training and testing
# test_size percentage of data gets tested on (0.9 training)

# lreg = LogisticRegression()
# lreg.fit(xtrain_bow, ytrain)    
# train using training dataset and label vector


# tfidf feature matrix
train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]


# ytrain.index index of axis labels 
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

# extract label vectors from the feature matrix via matching index

# print(xtrain_tfidf)


# fit on logistic regression
# lreg.fit(xtrain_tfidf, ytrain) 

# SVM
svm = SVC(kernel='rbf')
# Build using bow
svm.fit(xtrain_bow, ytrain)        
# accuracy test
svm.score(xvalid_bow, yvalid) 
# Build using TD-IDF
svm.fit(xtrain_tfidf, ytrain)      
svm.score(xvalid_tfidf, yvalid)
