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
# structure: word_id: word

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
dictionary.save('kaggle_dict')

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
'''
structure: [ [(word_id: num_times_occured), (), ...], ...]
with entry i in bow_corpus being the collection of words from document i

Using dictionary and bow_corpus together:

	bow_doc[i][j][0]: word id j from document i
	dictionary[bow_doc[i][j]]: actual word from document i
	bow_doc[i][j][0]: count of that word
'''

tfidf = models.TfidfModel(bow_corpus) #fit model on entire corpus

corpus_tfidf = tfidf[bow_corpus]

'''
tfidf score for each word in entire corpus

structure: [ [(word_id, tfidf_score], ...], ...]
where each entry in list is a document, each entry in entry is a word of that document
'''

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=30, id2word=dictionary, passes=2, workers=4)
lda_model_tfidf.save('kaggle_lda_tfidf')