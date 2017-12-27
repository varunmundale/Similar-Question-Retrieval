
from nltk.corpus import stopwords
# from autocorrect import spell


swords=stopwords.words('english')


def printList(lst):
	for item in lst:
		print item


def printDict(dct):
	for key in dct:
		print key,":",dct[key]

def writeDict(file,dct):
	f=open(file,'w')
	for key in dct:
		f.write(str(key)+":"+str(dct[key])+"\n\n");

	f.close()


def print2DDict(dct):
	for key in dct:
		inDict=dct[key]
		tuples=sorted(inDict.items(), key=lambda x: x[1])[::-1]
		print key
		for tup in tuples:
			print ":",tup,

		print 
		print "--------------------------------------------------------------------------------------"


def print2DDictOfWords(dct):
	for key in dct:
		inDict=dct[key]
		tuples=sorted(inDict.items(), key=lambda x: x[1])[::-1]
		print key

		count = 0
		for tup in tuples:
			if count <= 50:
				if tup[0] not in swords:
					print ":",tup,
					count+=1
			else:
				break;

		print 
		print "--------------------------------------------------------------------------------------"

def write2DDict(file,dct):
	f=open(file,'w')
	for key in dct:
		inDict=dct[key]
		tuples=sorted(inDict.items(), key=lambda x: x[1])[::-1]
		f.write(str(key)+"\n")
		for tup in tuples:
			f.write(":"+str(tup))

		f.write("\n\n--------------------------------------------------------------------------------------\n\n")

	f.close()

def printEmails(docs,labels):
	for doc,label in zip(docs,labels):
		print label
		print doc
		print "--------------------------------------------------------------------------------------------\n\n"