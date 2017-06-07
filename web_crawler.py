from urlparse import urlparse
import nltk
import urllib
import re
import requests
import lassie
import pprint
import csv
import unicodedata
import httplib
from nltk.corpus import stopwords
from nltk.stem.porter import *
from bs4 import BeautifulSoup
from collections import defaultdict

columns = defaultdict(list)
stemmer = PorterStemmer()

def remove_stopwords(tokens):
	filtered = [w for w in tokens if not w in stopwords.words('english')]
	return filtered

def stemming_word(tokens,stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed


def tokenize(text):
	tokenize_words = nltk.word_tokenize(text)
	stop_words = remove_stopwords(tokenize_words)
	stems = stemming_word(stop_words,stemmer)
	print stems
	return stems

def set_to_store():
	with open('timeset_demon.csv') as f:
		reader = csv.DictReader(f)
		for row in reader:
			for (k,v) in row.items():
				columns[k].append(v)

	

def checkUrl(url):
    p = urlparse(url)
    #print url
    conn = httplib.HTTPConnection(p.netloc)
    conn.request('HEAD', p.path)
    resp = conn.getresponse()
    #print resp.status
    return resp.status < 400 


def scrap_articles():
	i = 0
	j = 0
	set_to_store()
	for url,time in zip(columns['url'],columns['time']):
		j = j + 1
		#print j
		if j > 4824:
			print url
			url = url.strip()
			valid = checkUrl(url)
			if valid:
				page = requests.get(url)
				soup = BeautifulSoup(page.content,'html.parser')
				try:
					res = soup.findAll("div", {"id" : re.compile('content-body-14269002-[0-9]*')})
					if res:
						article_text = ''
			    		article = soup.find("div", {"id" : re.compile('content-body-14269002-[0-9]*')}).findAll('p')
			    		for element in article:
			       			article_text += ''.join(element.findAll(text = True))	
			       		article_text=tokenize(article_text)
			       		articles = u" ".join(article_text).encode('utf-8')
			       		li = []
			       		li.append(j)
			       		li.append(time)
			       		li.append(url)
			       		li.append(soup.title.string.strip().encode('utf-8'))
			       		li.append(articles)
			       		#print articles
			       		z = []
			       		z.append(li)
			       		"""
			       		with open('datasets1.csv','a') as fi:
			       			wr = csv.writer(fi)
			       			for val in z:
			       				wr.writerow(li)
			       			print i
			       			i = i + 1
			       		"""


				except(AttributeError, KeyError):
					pass


			break		   
		       	
def time_scrap():

	i = 1
	with open('dataset.csv','r') as f:
		for url in f:
			url = url.strip()
			if i > 12090:
				htmltext = urllib.urlopen(url).read()
				soup = BeautifulSoup(htmltext,'html.parser')
				l = []
				for tag in soup.find("span", {"class" : "blue-color ksl-time-stamp"}):
					t = soup.find('none')
					l.append(t.getText().strip())
					l.append(url)
					pprint.pprint(t.getText())
				z = []
				z.append(l)
				with open('timeset_demon.csv','a') as f:
					wtr = csv.writer(f)
					for val in z:
						wtr.writerow(val)
						#wtr.writecols(url)
				
			i = i + 1
			print i
		


def scrap():
	
	for i in range(1,2):
		url = "http://www.thehindu.com/search/?order=ASC&page="+`i`+"&q=demonetisation&sort=publishdate"
		htmltext = urllib.urlopen(url).read()
		soup = BeautifulSoup(htmltext,'html.parser')
		l = []
		for tag in soup.findAll("a", { "class" : "story-card75x1-text" }):
			l.append(tag['href'])
			pprint.pprint(tag['href'])
		with open('dataset.csv','a') as f:
			wtr = csv.writer(f)
			wtr.writerows(l)

def scrap_1():
	
	for i in range(1,2):
		url = "https://query.nytimes.com/search/sitesearch/?action=click&contentCollection&region=TopBar&WT.nav=searchWidget&module=SearchSubmit&pgtype=Homepage#/*/365days/document_type%3A%22article%22/"+`i`+"/allauthors/newest/U.S./"
		htmltext = urllib.urlopen(url).read()
		soup = BeautifulSoup(htmltext,'html.parser')
		productLinks = [div.a for div in soup.findAll('div', attrs={'class' : 'element2'})]
		print productLinks
    		
		
		#for tag in soup.findAll("div", { "class" : "element2" }):
		#	pprint.pprint(tag['href'])
		
"""		


"""

scrap_articles()
