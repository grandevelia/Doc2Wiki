import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests, json, os, sys

import pymysql.cursors
import pymysql
import pandas as pd

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='grandevelia',
                             password='Fr3ncht04$T',
                             db='signal',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor,
                             unix_socket='/Applications/XAMPP/xamppfiles/var/mysql/mysql.sock')

results = []
try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT content.url, topics.title FROM content JOIN content_topics ON content.id = content_topics.content_id JOIN topics ON content_topics.topic_id = topics.id"
        cursor.execute(sql)
        results = cursor.fetchall()
finally:
    connection.close()

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def find_body(text):
	return text
def find_top_n_sentences(text, n_sentences):
	return find_body(text)

training_data = []
for result in results:
	html = urllib2.urlopen(result['url'])
	article = find_top_n_sentences(text_from_html(html))
	tokenized_article = [w for w in word_tokenize(article.lower()) if w not in stopwords.words('english') and w.numeric()]