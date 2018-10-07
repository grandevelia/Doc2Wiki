import gensim
from gensim.models import Doc2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import os, sys
import random
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
#import urllib2.request


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

link = "https://www.nytimes.com/2018/06/12/opinion/earth-will-survive-we-may-not.html"
html = requests.get(link).text
article=text_from_html(html)

tokenized_article = [w for w in word_tokenize(article.lower()) if w not in stopwords.words('english') and w.isalpha()]
model = Doc2Vec.load('test.gensim')
for word in tokenized_article:
    tokenized_article = filter(lambda x: x in model.wv.vocab, tokenized_article)
'''
def find_body(text):
	return text
def find_top_n_sentences(text, n_sentences):
	return find_body(text)

article = find_top_n_sentences(text_from_html(article))
'''


print(model.wv.most_similar(tokenized_article, topn=10))
'''
ivec = model.infer_vector(tokenized_article)  
print(model.wv.most_similar([ivec],topn=3))
 '''