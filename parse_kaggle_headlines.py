import pandas as pd
import numpy as np

import gensim, nltk
from gensim import corpora, models
from nltk.stem.porter import *

from nlp_utils import lemmatize_stemming, preprocess

np.random.seed(42)

data = pd.read_csv('data/abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

nltk.download('wordnet')

processed_docs = documents['headline_text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
dictionary.save('kaggle_dict')

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=30, id2word=dictionary, passes=2, workers=4)
lda_model_tfidf.save('kaggle_lda_tfidf')