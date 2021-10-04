# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:43:43 2021

@author: sjhan
"""

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
import numpy as np
from multiprocessing import process, freeze_support
import nltk
from tempfile import mkdtemp
import os
import joblib


nltk.download('wordnet')
np.random.seed(2018)


def lemmatize_stemming(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    token = lemmatizer.lemmatize(text, pos='v')
    stem_token = stemmer.stem(token)
    return stem_token


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def get_dictionary_processed_docs():
    data = pd.read_csv('Top 500 Songs.csv',
                       error_bad_lines=False, encoding='mac_roman')
    data_text = data[['description']]
    data_text['index'] = data_text.index
    documents = data_text

    doc_sample = documents[documents['index'] == 430].values[0][0]

    processed_docs = documents['description'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=400)

    return dictionary, processed_docs


def main():

    dictionary, processed_docs = get_dictionary_processed_docs()
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    freeze_support()
    lda_model = gensim.models.LdaMulticore(
        bow_corpus, num_topics=20, id2word=dictionary, passes=2, workers=2)

    savedir = mkdtemp()
    filename = os.path.join(savedir, 'ldamodel.joblib')

    joblib.dump(lda_model, filename)
    print('Model trained and saved successfully')

    return filename


if __name__ == "__main__":
    main()