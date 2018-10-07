from xml.etree import ElementTree as ET
import codecs
import re
import json
import os, sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

filename = '/Applications/XAMPP/htdocs/learning_tests/data/simplewiki-20170201-pages-articles-multistream/simplewiki-20170201-pages-articles-multistream.xml'
#filename = '/Applications/XAMPP/htdocs/learning_tests/data/simplewiki-20170201-pages-articles-multistream/wiki_small.xml'
tree = ET.parse(filename)
root = tree.getroot()
path = os.path.abspath(os.curdir) + '/articles_corpus/'
url  = '{http://www.mediawiki.org/xml/export-0.10/}page'


def write_text_to_file(title, article_txt, i):
    outfile = path + str(i) +"_article.txt"
    #outfile = path + "test" + "_article.txt"
    article_json = dict([("title", title), ("text", article_txt)])
    article_json = json.dumps(article_json)
    with open(outfile, 'a+') as f:
    	json.dump(article_json, codecs.getwriter('utf-8')(f), ensure_ascii=False)

def clean_text(article_txt):
	for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
		article_txt = article_txt.replace(char, ' ' + char + ' ')
	# remove text written between double curly braces
	article_txt = re.sub(r"{{.*}}","",article_txt)
	article_txt = re.sub(r"\[\[File:.*\]\]","",article_txt)

	# remove Image attachments
	article_txt = re.sub(r"\[\[Image:.*\]\]","",article_txt)

	# remove unwanted lines starting from special characters
	article_txt = re.sub(r"^:\'\'.*","",article_txt)
	article_txt = re.sub(r"http\S+","",article_txt)
	article_txt = re.sub(r"\d+","",article_txt)
	article_txt = re.sub(r"\(.*\)","",article_txt)
	article_txt = re.sub(r"Category:.*","",article_txt)
	article_txt = re.sub(r"\| .*","",article_txt)
	article_txt = re.sub(r"\n\|.*","",article_txt)
	article_txt = re.sub(r".* \|\n","",article_txt)
	article_txt = re.sub(r".*\|\n","",article_txt)
	article_txt = re.sub(r"{{Infobox.*","",article_txt)
	article_txt = re.sub(r"{{infobox.*","",article_txt)
	article_txt = re.sub(r"{{taxobox.*","",article_txt)
	article_txt = re.sub(r"{{Taxobox.*","",article_txt)
	article_txt = re.sub(r"{{ Infobox.*","",article_txt)
	article_txt = re.sub(r"{{ infobox.*","",article_txt)
	article_txt = re.sub(r"{{ taxobox.*","",article_txt)
	article_txt = re.sub(r"{{ Taxobox.*","",article_txt)

	# remove text written between angle bracket
	article_txt = re.sub(r"<.*>","",article_txt)
	article_txt = re.sub(r"\n","",article_txt)
	article_txt = re.sub(r"\"|\#|\$|\%|\&|\(|\)|\*|\+|\-|\.|\/|\<|\=|\>|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~"," ",article_txt)

	# replace consecutive multiple space with single space
	article_txt = re.sub(r" +"," ",article_txt)

	# replace non-breaking space with regular space 
	article_txt = article_txt.replace(u'\xa0', u' ')
	tokenized_article = [w for w in word_tokenize(article_txt.lower()) if w.isalpha()]
	return " ".join(tokenized_article)

i = 0
for page in root.findall(url):
	for p in page:
		r_tag = "{http://www.mediawiki.org/xml/export-0.10/}revision"
		t_tag = "{http://www.mediawiki.org/xml/export-0.10/}title"
		if p.tag == t_tag:
			title = p.text
			check = re.match(r'^Category:', title)
			if check:

				break
			check = re.match(r'^Wikipedia:', title)
			if check:
				
				break
			check = re.match(r'^Template:', title)
			if check:
				
				break
			if title.isdigit():
				
				break


		if p.tag == r_tag:
			for x in p:
				tag = "{http://www.mediawiki.org/xml/export-0.10/}text"                               
				if x.tag == tag:                                                              
					text = x.text                                  
					if not text == None:
						i += 1
						text = clean_text(text)
						write_text_to_file(title, text, i)
        








