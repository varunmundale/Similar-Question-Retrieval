from __future__ import unicode_literals
import re
import numpy as np
# from printAndWriteMethods import *
import glob
import sys
import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from extract_features import *
import xml.etree.ElementTree as et

reload(sys)  
sys.setdefaultencoding('utf-8')



def createTrain(questions,answers):
	N=len(answers)
	#print np.shape(questions)
	X1=[]
	X2=[]
	Y=[]
	for i in tqdm(range(N)):
		question,answer=(questions[i],answers[i])
		#print np.shape(question)
		X1.append(question)
		X2.append(answer)
		Y.append(1)
		# print question,'===============>',answer,'     (Relevant)'
		question,answer=(questions[i],answers[random.randint(0,N-1)])
		# print question,'===============>',answer,'     (Irrelevant)'
		X1.append(question)
		X2.append(answer)
		Y.append(0)

	# sys.exit()
	return X1,X2,Y

def getQuestions(file):
	questions=[]
	f=open(file,'r')
	for line in f:
		question=line.split('\t')[2]
		questions.append(question)
	f.close()
	return questions

def getAnswers(file):
	answers=[]
	f=open(file,'r')
	for line in f:
		# print line
		answer='a'
		if len(line.split('|`|') )>0 and  len(line.split('|`|')[0].split('\t') )>1:
			answer=line.split('|`|')[0].split('\t')[1]	

		answers.append(answer)
	f.close()
	return answers

def getQuoraData():
	print '\nloading Quora '
	df = pd.read_csv('../data_quora/quora_duplicate_questions_small.tsv',sep='\t')

	df['question1']= df['question1'].apply(lambda x: unicode(x))
	df['question2']= df['question2'].apply(lambda x: unicode(x))
	# df['<label>']= df['<label>'].apply(lambda x: unicode(x))

	no_train=3000
	q1= list(df['question1'])
	q2 = list(df['question2'])

	q1=q1[:no_train]
	q2=q2[:no_train]
	
	labels= list(df['is_duplicate'])
	labels=labels[:no_train]

	print len(q1)

	return q1,q2,labels

def getTestData():
	print '\nloading Zhang test'
	df = pd.read_csv('../data/test_data/YahooData/yahoo_test.data',sep='\t')

	df['<query>']= df['<query>'].apply(lambda x: unicode(x))
	df['<candidate question>']= df['<candidate question>'].apply(lambda x: unicode(x))
	# df['<label>']= df['<label>'].apply(lambda x: unicode(x))

	no_test=5000
	q1= list(df['<query>'])
	q2 = list(df['<candidate question>'])

	q1=q1[:no_test]
	q2=q2[:no_test]
	
	labels= list(df['<label>'])
	labels=labels[:no_test]

	print len(q1)

	return q1,q2,labels



def getWebScopeQA():
	path='../Webscope_L5/manner-v2.0/manner.xml' 
	tree = et.parse(path)
	root = tree.getroot() 
	questions=[]
	answers=[]
	for vesp in tqdm(root.findall('vespaadd')):       
		for doc in vesp.findall('document'):
			for ques in doc.findall('subject'):
				questions.append(ques.text)	
				# print 'question:\n',ques.text,'\n'
			for ans in doc.findall('bestanswer'):
				answers.append(ans.text) 


	# 			# print 'answer:\n',ans.text,'\n\n\n\n\n'
	# tree = et.parse(path)
	# root = tree.getroot() 
	
	# for vesp in tqdm(root.findall('vespaadd')):       
	# 	for doc in vesp.findall('document'):
	# 		for ans in doc.find('bestanswer'):
	# 			answers.append(ans.text) 	
	no_samples=20000
	return questions[:no_samples],answers[:no_samples]



def getStackQA():
	path='../data/physics.stackexchange.com/Posts.xml'
	tree = et.parse(path) 
	root= tree.getroot()
	answer_q_dict={}
	question_dict={}
	answer_dict={}

	for row in root:
		if row.attrib['PostTypeId'] == '1' and 'AcceptedAnswerId' in row.attrib:    
			aid = row.attrib['AcceptedAnswerId']
			qid = row.attrib['Id']
			answer_q_dict[aid]= qid
			question_dict[qid]= row.attrib['Body']

	for row in root:
		if row.attrib['PostTypeId'] == '2' and row.attrib['Id'] in answer_q_dict:   
			aid=row.attrib['Id']
			qid = answer_q_dict[aid]
			answer_dict[qid] = row.attrib['Body']

	questions = [question_dict[qid] for qid in answer_dict]
	answers = [answer_dict[qid] for qid in answer_dict]
	no_samples=20000
	return questions[:no_samples],answers[:no_samples]

def getXYSplit():
	q1,q2,labels = getTestData()
	qq1,qq2,llabels = getQuoraData()

	print 'loading training'
	questions=[]
	answers=[]
	df_tr = pd.DataFrame() 

	# no_file=20
	# for i in range(no_file):
	# 	file='../data/train_data/YahooData/C'+str(i)+'Question.dat'
	# 	questions+=getQuestions(file)

	# for i in range(no_file):

	# 	file='../data/train_data/YahooData/C'+str(i)+'Answer.dat'
	# 	answers+=getAnswers(file);	


	questions,answers=getWebScopeQA()
	#questions,answers=getStackQA()

	print len(questions)
	print len(answers)
	
	# for q,a in zip(questions,answers):
	# 	print q,'===============>',a

	#sys.exit()
	df_tr['<question>']=questions
	df_tr['<answer>']=answers
		
	#df_test = pd.read_csv("../data/test_data/YahooData/yahoo_test.data",delimiter='\t')

	df_tr['<question>'] = df_tr['<question>'].apply(lambda x: unicode(x))
	df_tr['<answer>'] = df_tr['<answer>'].apply(lambda x: unicode(x))

	questions=list(df_tr['<question>'])
	answers=list(df_tr['<answer>'])


	trigramsDict=populateTrigramDict(questions+answers+q1+q2+qq1+qq2)
	# trigramsDict=populateTrigramDict(questions+answers+q1+q2)
	questions,answers,Y=createTrain(questions,answers)

	
	q_vectors=featureVectors(trigramsDict,questions)
	a_vectors=featureVectors(trigramsDict,answers)
	q1_vectors = featureVectors(trigramsDict,q1)
	q2_vectors = featureVectors(trigramsDict,q2)
	qq1_vectors = featureVectors(trigramsDict,qq1)
	qq2_vectors = featureVectors(trigramsDict,qq2)
	
	del questions
	del answers
	del q1
	del q2

	X1_train=np.array(q_vectors+qq1_vectors)
	X2_train=np.array(a_vectors+qq2_vectors)
	Y_train=np.array(Y+llabels)
	X1_test=np.array(q1_vectors)
	X2_test=np.array(q2_vectors)
	Y_test=np.array(labels)

	del q_vectors
	del a_vectors

	#print X1.shape,X2.shape,Y.shape#,X1_test.shape,X2_test.shape,Y_test.shape
	# df = pd.DataFrame() 

	# df['<q_feats>']=list(X1)
	# df['<a_feats>']=list(X2)
	# df['<labels>']=list(Y)

	# df = df.reindex(np.random.permutation(df.index))
	# num_train = int(X1_train.shape[0] * 1)

	# X1,X2,Y=df['<q_feats>'],df['<a_feats>'],df['<labels>']
	# X1,X2,Y=np.array(X1),np.array(X2),np.array(Y)

	# X1.reshape(len(X1),len(X1[0]))
	# print X1.shape
	# print 'sum tri',sum(X1[1])#,len(X1[0])
	# exit();
	size=len(X1_train)
	h=1
	w=len(X1_train[0])
	d=1
	X1_train=X1_train.reshape(size,h,w,d)
	X2_train=X2_train.reshape(size,h,w,d)

	size=len(X1_test)
	w = len(X1_test[0])

	X1_test=X1_test.reshape(size,h,w,d)
	X2_test=X2_test.reshape(size,h,w,d)

	# X1_train=X1[:num_train]
	# X2_train=X2[:num_train]
	# Y_train=Y[:num_train]

	# X1_test=X1[num_train:]
	# X2_test=X2[num_train:]
	# Y_test=Y[num_train:]
	# X1_train,X2_train,Y_train,X1_test,X2_test,Y_test=np.array(X1_train),np.array(X2_train),np.array(Y_train),np.array(X1_test),np.array(X2_test),np.array(Y_test)
	print X1_train.shape,X2_train.shape,Y_train.shape,X1_test.shape,X2_test.shape,Y_test.shape
	
	return X1_train,X2_train,Y_train,X1_test,X2_test,Y_test

# getXYSplit()
