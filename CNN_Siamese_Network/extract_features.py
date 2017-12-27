from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer
import sys
import pickle

reload(sys)
sys.setdefaultencoding('utf8')

stopWords = set(stopwords.words('english'))
stemmer=PorterStemmer()
dict_stems={}


def caseFolding(text):
	return text.lower()

def cleanText(text):
	if text==None:
		return ""

	text=caseFolding(text)
	# text=cleanTags(text)
	return text

def removeStopWords(tokens):	
	return [token for token in tokens if token not in stopWords];

def getStemmedTokens(text):
	global 	dict_stems
	if text==None:
		return [];
	tokens=[]
	text= re.sub(u'[^a-zA-Z0-9]+', ' ',text)
	tokens=tokens+text.split()
	tokens=[token.strip() for token in tokens]

	for i,token in enumerate(tokens):
		if len(token)>50:
			continue;
		if token not in dict_stems:
			s_tok=stemmer.stem(token) if token.isalpha() else token
			dict_stems[token]=s_tok
		else:
			s_tok=dict_stems[token]#retrieve stemmed

		tokens[i]=s_tok


	return tokens



def getLetterTrigrams(sent):
	sent=cleanText(sent)
	tokens=getStemmedTokens(sent)
	tokens=removeStopWords(tokens)
	trigrams={}

	for t in tokens:
		t='#'+t+'#'
		for i in range(2,len(t)):
			trigram=t[i-2]+t[i-1]+t[i]
			if trigram not in trigrams:
				trigrams[trigram]=0
			trigrams[trigram]+=1
	return trigrams


def populateTrigramDict(sentences):
	print 'populating trigram dict.......'
	trigramsDict={}
	for sent in sentences:
		sent=cleanText(sent)
		tokens=getStemmedTokens(sent)
		tokens=removeStopWords(tokens)
		for t in tokens:
			t='#'+t+'#'
			for i in range(2,len(t)):
				trigram=t[i-2]+t[i-1]+t[i]
				if trigram not in trigramsDict:
					trigramsDict[trigram]=0
					
	print 'Saving trigram dictionary'
	tri_out = open('trigramsDict.pkl', 'wb')			
	pickle.dump(trigramsDict, tri_out)
	tri_out.close()

	return trigramsDict


def featureVectors(trigramsDict,sentences):
	print("Dict len:",len(trigramsDict))
	vectors=[]
	for sent in tqdm(sentences):
		trigs = getLetterTrigrams(sent)
		trigs= list(trigs.keys())

		vector=np.zeros(len(trigramsDict))
		indexTri={}
		for i,key in enumerate(trigramsDict):
			indexTri[key]=i;

		for t in trigs:
			# print(t)
			if t in indexTri:
				vector[indexTri[t]]+=1
		vectors.append(vector)
		# print sum(vector)	
	return vectors

# sentences= ['where are you now?','songs of tomorrow and dawn will follow']
# populateTrigramDict(sentences)
# print(featureVectors(sentences))
