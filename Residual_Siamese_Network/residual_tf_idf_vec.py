# avoid decoding problems
import sys
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
import xml.etree.ElementTree as et
import random

reload(sys)  
sys.setdefaultencoding('utf-8')
df= pd.DataFrame()
nlp = spacy.load('en')


def getTestData():
	print '\nloading Zhang test'
	df = pd.read_csv('../data/test_data/YahooData/yahoo_test.data',sep='\t')

	df['<query>']= df['<query>'].apply(lambda x: unicode(x))
	df['<candidate question>']= df['<candidate question>'].apply(lambda x: unicode(x))
	# df['<label>']= df['<label>'].apply(lambda x: unicode(x))

	no_test=10000
	q1= list(df['<query>'])
	q2 = list(df['<candidate question>'])

	q1=q1[:no_test]
	q2=q2[:no_test]
	
	labels= list(df['<label>'])
	labels=labels[:no_test]

	print len(q1)

	return q1,q2,labels

def getFeatures(df,col):
	vecs1 = [doc.vector for doc in nlp.pipe(df[col], n_threads=50)]
	vecs1 =  np.array(vecs1)
	# print 'shape:',vecs1.shape
	return vecs1

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

if __name__ == "__main__":
	########## READ DATA #########
	print("Reading Data")
	df = pd.read_csv("data/quora_duplicate_questions_small.tsv",delimiter='\t')
	questions,answers=getStackQA()
	questions,answers,labels=createTrain(questions,answers)
	q = df['question1']
	a = df['question2']
	labels = np.array(list(df['is_duplicate']) + labels)
	df= pd.DataFrame()
	df['question1']=list(q)+questions
	df['question2']=list(a)+answers

	# encode questions to unicode
	df['question1'] = df['question1'].apply(lambda x: unicode(x))
	df['question2'] = df['question2'].apply(lambda x: unicode(x))

	###### exctract word2vec vectors ######
	print("Converting to word Vectors")


	df['q1_feats'] = getFeatures(df,'question1')
	df['q2_feats'] = getFeatures(df,'question2')

	# save features
	# pd.to_pickle(df, 'data/1_df.pkl')


	####### shuffle df ########
	# df = df.reindex(np.random.permutation(df.index))

	# set number of train and test instances
	num_train = int(df.shape[0] * 1)
	print("Number of training pairs: %i"%(num_train))


	# init data data arrays
	X_train = np.zeros([num_train, 2, 300])
	Y_train = np.zeros([num_train]) 

	# format data 
	b = [a[None,:] for a in list(df['q1_feats'].values)]
	q1_feats = np.concatenate(b, axis=0)

	b = [a[None,:] for a in list(df['q2_feats'].values)]
	q2_feats = np.concatenate(b, axis=0)

	# fill data arrays with features
	X_train[:,0,:] = q1_feats[:num_train]
	X_train[:,1,:] = q2_feats[:num_train]
	# Y_train = df[:num_train]['is_duplicate'].values
	Y_train=labels
	# X_test[:,0,:] = q1_feats[num_train:]
	# X_test[:,1,:] = q2_feats[num_train:]
	# Y_test = df[num_train:]['is_duplicate'].values
	print 'shape train:', np.shape(X_train),np.shape(Y_train)

	del b
	del q1_feats
	del q2_feats


	df= pd.DataFrame()
	q1,q2,labels = getTestData()
	num_test= len(q1)
	print("Number of testing pairs: %i"%(num_test))
	X_test  = np.zeros([num_test, 2, 300])
	# Y_test = np.zeros([num_test])


	df['question1']=q1
	df['question2']=q2

	df['q1_feats'] = getFeatures(df,'question1')
	df['q2_feats'] = getFeatures(df,'question2')


	b = [a[None,:] for a in list(df['q1_feats'].values)]
	q1_feats = np.concatenate(b, axis=0)

	b = [a[None,:] for a in list(df['q2_feats'].values)]
	q2_feats = np.concatenate(b, axis=0)

	# fill data arrays with features
	X_test[:,0,:] = q1_feats[:]
	X_test[:,1,:] = q2_feats[:]
	Y_test = np.array(labels)

	print 'shape test:', np.shape(X_test),np.shape(Y_test)





	# remove useless variables
	del b
	del q1_feats
	del q2_feats

	####### create model ############
	print("Creating model")

	net = create_network(300)

	print("Training")
	# train
	#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
	optimizer = Adam(lr=0.001)
	net.compile(loss=contrastive_loss, optimizer=optimizer)

	for epoch in range(50):
	    net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
	          validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
	          batch_size=128, nb_epoch=1, shuffle=True, )
	    
	    # compute final accuracy on training and test sets
	    pred = net.predict([X_test[:,0,:], X_test[:,1,:]], batch_size=128)
	    te_acc = compute_accuracy(pred, Y_test)
	    
	#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
	    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))




print 'Saving model...'
net_json=net.to_json()
model_out = open('model_json.pkl', 'wb')			
pickle.dump(net_json, model_out)
model_out.close()
net.save_weights('model_weights.h5')

print 'Done'