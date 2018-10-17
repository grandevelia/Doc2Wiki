import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

   
stemmer = SnowballStemmer("english") 
#link = "https://www.nytimes.com/2018/06/12/opinion/earth-will-survive-we-may-not.html"
#link = "https://www.slowtwitch.com/Products/Tri_Bike_by_brand/Specialized/S-Works_Shiv_Disc_7053.html"
article = "Cow horse pig chicken farm tractor milk"
#html = requests.get(link).text
#article = text_from_html(html)

lda_model_tfidf = LdaMulticore.load("kaggle_lda_tfidf")
dictionary = Dictionary.load('kaggle_dict')

bow_vector = dictionary.doc2bow(preprocess(article))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

