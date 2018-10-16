import pandas as pd
import numpy as np

import gensim, nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

np.random.seed(42)

data = pd.read_csv('data/abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

stemmer = SnowballStemmer("english")

nltk.download('wordnet')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = documents['headline_text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=30, id2word=dictionary, passes=2, workers=4)

unseen_document = "I began the Kona Bike Survey in 1992, and ran it through 2006, at which point I gladly handed it off to the next fool. But I, like you, am interested in the results. Here they are, for those who are interested: [The chart above omits certain brands like Argon 18 (terrific brand, 118 bikes this year), and perhaps others, because they weren't factors on the chart below, which is my focus at the moment.] Cervelo had the most bikes on the pier. What that brand has done over the last dozen years is stupendous. But I was only marginally interested in that particular metric as a bike brand owner. I wanted to know more how I did year over year, ex any changes in the sponsored bikes on the pier. What I found in my years doing the survey is that there was a carryover minus attrition, and I could never get that completely dialed for every brand but it worked something like this: If a brand completely disappeared. Blew up. Factory blew up. All bikes in the pipeline vanished. People would continue to own the bikes they owned. They would continue to qualify for Kona and race their bikes. That was the carryover. The attrition was, maybe, 20 percent plus or minus? Some percentage of those people didn’t race Kona the next year, replaced by others (with other bike brands underneath them). Some returned, but with new bikes. That attrition rate may have been 25 percent, but I think it’s shrunk, for two reasons: First, we have professional amateurs now. Much higher rate of requalifiers. Second, bikes got expensive! And they got good. They last longer. They need to be replaced less often. And, the incremental gains are less, year over year, in quality, so, again, fewer reasons to replace them. Who 'won' the Kona survey this year, in 2018? If the metric is “net” increases? One could fault my methodology and I invite critiques. My formula is this: reduce 2017 totals by 15 percent, and compare that to 2018 totals. If you look at it simply on this basis, here are the “winners”:Using this model, Canyon is the “winner” because it survived its attrition and added a lot of bikes. Felt is 2nd, Trek 3rd, QR 4th. Does this pass the test of reasonableness? Cervelo remained a hot brand, but I think it suffered a little through its flagship bike entry of last year being both very expensive and also difficult to produce in bulk. Canyon had the benefit of a year of sales in the U.S. Specialized suffered, having a very small net increase, but its bikes were quite long in the tooth (the Shiv Disc was originally schedule to be out last season, and it won’t be out until April). I’ll be interested to see how the IBDs feel about the analysis above. Did Trek and Felt dealers feel they had a good year? I know Quintana Roo (my old job!) is bullish on its 2018 progress. Canyon? It sold a lot of bikes in the U.S. in 2018, but not nearly the tri bikes it could have sold (forecasting’s a biatch). Next year? Canyon’s going to be tough to beat, using my formula, because it’ll right its forecasting ship in the U.S. Felt just produced a new bike. Specialized? I don’t think it’ll make much headway unless it can do what Felt just did and downstream its new flagship bike in a hurry. Even Felt is going to need to bring disc brakes down to the IAx level before it really takes off. QR has been money this year, because it's got disc brake tri bikes down to $4,500. But it's game on to get disc brake tri bikes down to $3,000."

bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))