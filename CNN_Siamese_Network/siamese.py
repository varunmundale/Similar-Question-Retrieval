
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge, Conv1D, GlobalMaxPooling1D, Conv2D,GlobalMaxPooling2D, MaxPooling2D ,Flatten,dot
from keras import backend as K
from myPrint import *
from getQuestionAnswers import getTestData
import gensim
from sklearn.metrics import average_precision_score
import sys

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 128
kernel_size = 3
hidden_dims = 128
epochs = 2
alpha = 0.7


def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

# def cosine_distance(vests):
#     x, y = vests
#     x = K.l2_normalize(x, axis=-1)
#     y = K.l2_normalize(y, axis=-1)
#     return -K.mean(x * y, axis=-1, keepdims=True)

# def cos_dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0],1)


# def cosine_dist(vects):
# 	x,y=vects
# 	return K.dot(x,K.transpose(y))

def cosine_dist(vects):
	return dot(vects,axes=1,normalize=True)


def cos_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 0.0
#     return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def contrastive_loss(y_true, y_pred):
	margin = 0.5
	return K.mean( y_true * (1-y_pred) + (1 - y_true) * K.maximum(y_pred-margin, 0) )


# def create_base_network(input_dim):
# 	'''
# 	Base network for feature extraction.
# 	'''
# 	model = Sequential()
	
# # we add a Convolution1D, which will learn filters
# # word group filters of size filter_length:
# 	model.add(Conv2D(filters,
# 					 kernel_size=(1,3),
# 					 strides=(1,1), input_shape=(1,input_dim,1) )) 
# 	# we use max pooling:
# 	model.add(GlobalMaxPooling2D())
# 	model.add(Activation('relu'))

# 	# We add a vanilla hidden layer:
# 	model.add(Dense(hidden_dims))
# 	model.add(Dropout(0.2))

# 	print(model.summary())
# 	return model


def create_base_network(input_dim):
	# embeddingSize = 128
	# input_shape = (1, input_dim, 1)
	# kernel_size_conv1 = (1,10)
	# kernel_size_conv2 = (1,5)
	# pool_size1 = (1, 100)
	# pool_size2 = (1, 5)
	# pool_stride1 = (1, 100)
	# pool_stride2 = (1, 4)
	# out_conv1 = 5 
	# out_conv2 = 8 

	embeddingSize = 128
	input_shape = (1, input_dim, 1)
	kernel_size_conv1 = (1,10)
	kernel_size_conv2 = (1,5)
	
	pool_size1 = (1, 100)
	pool_size2 = (1, 5)
	
	pool_stride1 = (1, 100)
	pool_stride2 = (1, 4)
	
	out_conv1 = 5 
	out_conv2 = 8 
	
	model = Sequential()

	model.add(Conv2D(out_conv1, kernel_size = kernel_size_conv1, input_shape = input_shape))
	model.add(MaxPooling2D(pool_size = pool_size1, strides = pool_stride1))
	model.add(Activation('relu'))

	model.add(Conv2D(out_conv2, kernel_size = kernel_size_conv2))
	model.add(MaxPooling2D(pool_size = pool_size2, strides = pool_stride2))
	model.add(Activation('relu'))

	# model.add(Conv2D(out_conv3, kernel_size = kernel_size_conv3))
	# model.add(MaxPooling2D(pool_size = pool_size3, strides = pool_stride3))
	# model.add(Activation('relu'))
	
	model.add(Flatten())
	model.add(Dense(embeddingSize)) 
	# model.add(Activation('sigmoid'))

	return model

def get_bm25_score():
	
	q1,q2,labels = getTestData()
	bm25_scores=[]
	for i in range(len(q1)):
		bm25 = gensim.summarization.bm25.BM25(q1[i])
		average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
		score = bm25.get_scores(q2[i],average_idf)
		bm25_scores.append(score[0])
	return bm25_scores

def compute_accuracy(preds, labels):
	'''
	Compute classification accuracy with a fixed threshold on distances.
	'''
	bm25_scores = get_bm25_score()
	for i,pred in enumerate(preds):
		preds[i] = alpha*pred[0] + (1-alpha)*bm25_scores[i]
	print('labels.........')
	

	preds = [ preds[i] for i in range(len(preds)) ]
	preds1 = np.array([ float(preds[i][0]) for i in range(len(preds)) ])
	# print(np.shape(labels),np.shape(preds1))
	# print('predictions......')
	# printList(preds1)
	labels=([int(l) for l in labels])
	# print(labels.count(1) + labels.count(0))


	mean_ap =  average_precision_score(labels,preds1)

	avg = sum(preds)/float(len(preds))
	# printList(preds)
	# avg=avg-0.1
	print('average-------',avg)

	preds = [ 1 if preds[i]>= avg else 0 for i in range(len(preds))  ]

	# for p,l in zip(preds,labels):
	# 	print(p,l)


	acc = sum( [ 1 if labels[i]==preds[i] else 0 for i in range(len(labels)) ] ) /float(len(labels))
	
	return acc,mean_ap

def create_network(input_dim):
	# network definition
	base_network = create_base_network(input_dim)
	
	input_a = Input(shape=(1,input_dim,1))
	input_b = Input(shape=(1,input_dim,1))
	
	print('type',type(input_a))
	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)	
	distance = Lambda(cosine_dist, output_shape=cos_dist_output_shape)([processed_a, processed_b])
	# distance = Activation('sigmoid')(distance)
	model = Model(input=[input_a, input_b], output=distance)
	return model
