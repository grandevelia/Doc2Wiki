import requests

import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary

from '../../nlp_utils' import lemmatize_stemming, preprocess, text_from_html, tag_visible

#link = "https://www.nytimes.com/2018/06/12/opinion/earth-will-survive-we-may-not.html"
#link = "https://www.slowtwitch.com/Products/Tri_Bike_by_brand/Specialized/S-Works_Shiv_Disc_7053.html"

#html = requests.get(link).text
#article = text_from_html(html)
article = "How a Pentagon deal became an identity crisis for Google"

lda_model_tfidf = LdaMulticore.load("kaggle_lda_tfidf")
dictionary = Dictionary.load('kaggle_dict')

bow_vector = dictionary.doc2bow(preprocess(article))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

