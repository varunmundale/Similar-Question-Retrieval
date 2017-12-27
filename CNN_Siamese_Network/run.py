from __future__ import unicode_literals

# -*- coding: utf-8 -*-
'''
    It is really simple algorithm based on word2vec
    - convert mean word to vec representations of the questions
    - train a simple model for pairs and see the difference
'''
# avoid decoding problems
import sys
import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
from getQuestionAnswers import getXYSplit
from myPrint import *;
import pickle



reload(sys)  
sys.setdefaultencoding('utf-8')
X1_train,X2_train,Y_train,X1_test,X2_test,Y_test=getXYSplit()
          
# create model
from siamese import *
from keras.optimizers import RMSprop, SGD, Adam

print 'X1 width',X1_train.shape[2]
net = create_network(X1_train.shape[2])

# train
#optimizer = SGD(lr=0.01, momentum=0.05, nesterov=True, decay=0.004)
optimizer = SGD(lr=0.01, momentum=0.05)
# optimizer = Adam(lr=0.01)
net.compile(loss=contrastive_loss, optimizer=optimizer,metrics=['accuracy'])

for epoch in range(10):
    net.fit([X1_train,X2_train], Y_train,
          batch_size=50, nb_epoch=1, shuffle=True,validation_split=0.1 )
    
    # compute final accuracy on training and test sets
    pred = net.predict([X1_test,X2_test], batch_size=50)
    # printList(pred)
    # printList(Y_test)
    # print pred
    te_acc,mean_ap = compute_accuracy(pred, Y_test)
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc),mean_ap)

print 'Saving model...'
net_json=net.to_json()
model_out = open('model_json.pkl', 'wb')			
pickle.dump(net_json, model_out)
model_out.close()
net.save_weights('model_weights.h5')

print 'Done'

# print 'Saving Model...'
# model_out = open('model.pkl', 'wb')			
# pickle.dump(net, model_out)
# model_out.close()
