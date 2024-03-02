# imports
import collections
import math
import random
import re
import requests
import sklearn
import string
import urllib
# some NLP libraries that can help us with preprocessing
import nltk
# for coloring out outputs
from termcolor import colored
# for comparing strings
import difflib
import pandas as pd
from bs4 import BeautifulSoup
import preprocessor as tweet_preprocessor
from ttp import ttp
import emoji
import ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# Load the dataset
df = pd.read_csv("dataset_cl.csv")  # Replace "your_dataset.csv" with the actual path to your dataset
df = df.sample(frac=1,)
label_map = {'fake': 0, 'real': 1}
df['label'] = df['label'].map(label_map)
#init of componunts
stemmer = nltk.stem.porter.PorterStemmer()
stoplist = set(nltk.corpus.stopwords.words('english'))
tt = nltk.tokenize.TweetTokenizer()
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
punct_set = set(string.punctuation + '''…'"`’”“''')

# get a set of punctuation
from nltk.tokenize import word_tokenize
punct_set = set(string.punctuation + '''…'"`’”“'''  + '️')
# Use NLTK for tokenization
def nltk_tokenize(text):
    return tt.tokenize(text)
stoplist = set(stoplist)
def emoji_split(e, joiner = '\u200d',
                variation_selector=b'\xef\xb8\x8f'.decode('utf-8'),
                return_special_chars = False):
  parts = []
  for part in e:
    if part == joiner:
      if return_special_chars:
        parts.append(":joiner:")
    elif part == variation_selector:
      if return_special_chars:
        parts.append(":variation:")
    else:
      parts.append(part)
  return parts
def my_preprocessor(text, tokenize=nltk_tokenize,
                         hashtag_segmenter=hashtag_segmenter,
                         punct_set=punct_set, stoplist=stoplist):

  # lowercase
  text = text.lower()
  # tokenize
  tokens = tokenize(text)
  updated_tokens = []
  # set different behavior for different kinds of tokens
  for t in tokens:
    # split emoji into components
    if t in emoji.EMOJI_DATA:
      updated_tokens += emoji_split(t)
    # keep original hashtags and split them into words
    elif t.startswith('#'):
#       updated_tokens += [t]
      updated_tokens += hashtag_segmenter.pre_process_doc(t).split()
    # remove user mentions
    elif t.startswith('@'):
      pass
    # remove urls because we will get them from the expanded_urls field anyways
    # and remove single punctuation markers
    elif t.startswith('http') or t in punct_set:
      pass
    # skip stopwords and empty strings, include anything else
    elif t and t not in stoplist:
      updated_tokens += [t]
            
  return ' '.join(updated_tokens)

preprocessed_tweets = []

for index, row in df.iterrows():
    processed_tweet = my_preprocessor(row['tweet'])
    preprocessed_tweets.append(processed_tweet)

# Create a new DataFrame with preprocessed tweets
df_preprocessed = df.copy()
df_preprocessed['tweet'] = preprocessed_tweets

# Check the first few rows of the new DataFrame
print(df_preprocessed.head())
print(df.head())

from sklearn.model_selection import train_test_split

# Split the dataset into train, validation, and test sets
# train_val, test = train_test_split(df_preprocessed, test_size=0.1, shuffle=True, random_state=42)
# train, val = train_test_split(train_val, test_size=0.1111, shuffle=True, random_state=42)
train, test = train_test_split(df_preprocessed, test_size=0.2, shuffle=True, random_state=42)
test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=42)


# Dump the splits into CSV files
train.to_csv("train_preprocessed.csv", index=False)
val.to_csv("val_preprocessed.csv", index=False)
test.to_csv("test_preprocessed.csv", index=False)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
# tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data
tfidf_vectorizer.fit_transform(train['tweet'])

# Transform the training data into TF-IDF vectors
X_train_tfidf = tfidf_vectorizer.transform(train['tweet'])
X_validation_tfidf = tfidf_vectorizer.transform(val['tweet'])
X_test_tfidf = tfidf_vectorizer.transform(test['tweet'])

# Extract labels
y_train = train['label'].values
y_validation = val['label'].values
y_test = test['label'].values

# Check the shape of the resulting TF-IDF matrix
print("Shape of TF-IDF matrix:", X_train_tfidf.shape)
X_val_tfidf = tfidf_vectorizer.transform(val['tweet'])
# accuracy_knn = accuracy_score(val['label'], val_predictions_knn)
print("Shape of VAl TF-IDF matrix:", X_val_tfidf.shape)
print(y_test)
import pickle
with open('train_tfidf.pkl', 'wb') as f:
    pickle.dump(X_train_tfidf, f)

with open('validation_tfidf.pkl', 'wb') as f:
    pickle.dump(X_validation_tfidf, f)

with open('test_tfidf.pkl', 'wb') as f:
    pickle.dump(X_test_tfidf, f)

# Save the TfidfVectorizer instance
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save the labels
with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('y_validation.pkl', 'wb') as f:
    pickle.dump(y_validation, f)

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)