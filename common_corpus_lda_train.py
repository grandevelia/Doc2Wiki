import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary

import nltk
from nltk.stem.porter import *
from nlp_utils import lemmatize_stemming, preprocess

np.random.seed(42)
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

lda_model_tfidf = gensim.models.LdaMulticore(common_corpus, num_topics=10, id2word=common_dictionary, workers=4)
lda_model_tfidf.save('common_lda')