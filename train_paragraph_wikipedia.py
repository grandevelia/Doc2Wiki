import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import os, sys
import random
import math
import multiprocessing

path = os.path.abspath(os.curdir)
outfile = path + "/test_nums.txt"
#print outfile
#sys.exit()
random.seed(43)
num_articles = 173229
num_to_test = 100#num_articles-1
#train_size = 0.8
#test_floor = int(train_size * num_to_test)

file_nums = random.sample(xrange(1, num_articles), num_to_test)

def find_body(text):
	return text
def find_top_n_sentences(text, n_sentences):
	return find_body(text)

train_set = []
#test_set = []

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

i = 0;

for file_num in file_nums:
	filename = str(file_num) + "_article.txt"
	
	#if (i < test_floor):
	with open('articles_corpus/' + filename, 'r') as file:
		data = json.loads(json.load(file))
		article = data['text']
		title = data['title']

		tokenized_article = [w for w in word_tokenize(article.lower()) if w not in stopwords.words('english') and w.isalpha()]
		doc = TaggedDocument(tokenized_article, title)
		train_set.append(doc)
	'''
	else:
		test_set.append(file_num)
	'''

	i += 1
f = open(outfile, 'w')
'''
for num in test_set:
	f.write("%s\n" % num)
'''

model = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, workers=cores, epochs=20)
model.build_vocab(train_set)
model.train(train_set, total_examples=model.corpus_count, epochs=model.epochs)
model.save('small_test.gensim')

'''
article = find_top_n_sentences(text_from_html(article))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2,)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save('test')

ivec = model.infer_vector(doc_words=tokenized_article, steps=20, alpha=0.025)
print(model.most_similar(positive=[ivec], topn=10))'''