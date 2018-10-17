import gensim
from gensim.models import Doc2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import os, sys
import random

random.seed(43)
num_articles = 173229
num_to_test = 100
test_nums = random.sample(xrange(1, num_articles), num_to_test)

model = Doc2Vec.load('small_test.gensim')

correct = 0.0
total = 0.0

test_nums = [test_nums[0]]
for test_num in test_nums:
    with open("articles_corpus/" + str(test_num) + "_article.txt") as test_file:
        data = json.loads(json.load(test_file))
        article = data['text']
        title = data['title']

        tokenized_article = [w for w in word_tokenize(article.lower()) if w not in stopwords.words('english') and w.isalpha()]
        for word in tokenized_article:
            tokenized_article = filter(lambda x: x in model.wv.vocab, tokenized_article)

        print(title)
        model_out = model.wv.most_similar(tokenized_article, topn=10)
        print(model_out)
        if (title in model_out):
            correct = correct + 1.0

        total = total + 1.0

print("Accuracy: " + str(correct/total))
print("Correct: " + str(correct))
print("Total: " + str(total))