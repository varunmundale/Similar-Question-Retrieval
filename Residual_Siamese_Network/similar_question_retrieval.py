import sys
import numpy as np
from keras.models import model_from_json
from siamese import *
import pickle
from keras.optimizers import RMSprop, SGD, Adam
from myPrint import *
import spacy
import gensim
from sklearn.metrics import average_precision_score
from quora_spacy_tf_idf_vec import *

reload(sys)  
sys.setdefaultencoding('utf-8')

df= pd.DataFrame()


# dim=len(trigramsDict)
dim=300
net = create_network(dim)
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

net.load_weights('model_weights.h5')

# net_weights = load_model('cnn_model')

def getSimilar(X1_test,X2_test): 
	pred = net.predict([X1_test,X2_test], batch_size=50)

	return pred 

	# if pred[0]<=0.8:
	# 	print 'Dissimilar'
	# else:
	# 	print 'Similar.......'

def getQuestions():
	print '\nloading Zhang test'
	df = pd.read_csv('../data/test_data/YahooData/yahoo_test.data',sep='\t')
	df['<candidate question>']= df['<candidate question>'].apply(lambda x: unicode(x))
	c_questions=list(df['<candidate question>'])
	# c_ques_scores=[(q,0) for q in c_questions]

	return c_questions

def getQVectors(c_questions):
	n=len(c_questions)
	qVectors=[]
	for i in tqdm(range(n)):
		q2=c_questions[i]
		df['question2']=[q2]
		df['question2'] = df['question2'].apply(lambda x: unicode(x))
		q2_feats=getFeatures(df,'question2')
		X2_test=q2_feats
		# print 'shape:',X2_test.shape

		qVectors.append(X2_test)

	# qVectors=np.array(qVectors)
	print 'shape:',np.shape(qVectors)
	return qVectors

def get_bm25_score(q1):
	bm25_scores = bm25.get_scores(q1,average_idf)
	return bm25_scores

print 'Loading Questions ...'
c_questions=getQuestions()
# BM25
# bm25 = gensim.summarization.bm25.BM25([q.split() for q in c_questions])
# average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

n=len(c_questions)
print 'Creating Q Vectors ...'
qVectors = getQVectors(c_questions)

while(True):
	print '\n\n'
	q1=raw_input('Enter Question:')
	df['question1']=[q1]
	df['question1'] = df['question1'].apply(lambda x: unicode(x))
	q1_feats=getFeatures(df,'question1')

	# bm25_scores = get_bm25_score(q1)
	c_ques_scores=[]
	
	for i in tqdm(range(n)):
		q2=c_questions[i]
		q2_feats=qVectors[i]

		X1_test=np.array(q1_feats)
		X2_test=np.array(q2_feats)
		# print 'X1_test',np.shape(X1_test)
		# print 'X2_test',np.shape(X2_test)
		score=getSimilar(X1_test,X2_test)
		# score = alpha*score + (1-alpha)*bm25_scores[i]
		c_ques_scores.append( (q2,score) )

	# scores=[c[1] for c in c_ques_scores]
	# print 'Mean average precision score',average_precision_score()
	c_ques_scores=sorted(c_ques_scores,key=lambda x:x[1],reverse=True)

	print '\nSimilar Questions:\n'
	printList(c_ques_scores[:5])