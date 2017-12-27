import numpy as np
import pandas as pd


def getLetterTrigrams(sentence):
	tokens=sentence.split()
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
	trigramsDict={}
	for sent in sentences:
		tokens=sent.split()
		for t in tokens:
			t='#'+t+'#'
			for i in range(2,len(t)):
				trigram=t[i-2]+t[i-1]+t[i]
				if trigram not in trigramsDict:
					trigramsDict[trigram]=0
				trigramsDict[trigram]+=1

	return trigramsDict


def featureVectors(trigramsDict,sentences):
	print("Dict len:",len(trigramsDict))
	vectors=[]
	trigramsDictKeys=list(trigramsDict.keys())
	for sent in sentences:
		trigs = getLetterTrigrams(sent)
		trigs= list(trigs.keys())
		vector=np.zeros(len(trigramsDict))
		for t in trigs:
			# print(t)
			vector[trigramsDictKeys.index(t)]=trigramsDict[t]
		vectors.append(vector)
	return vectors

# sentences= ['where are you now?','songs of tomorrow and dawn will follow']
# populateTrigramDict(sentences)
# print(featureVectors(sentences))
