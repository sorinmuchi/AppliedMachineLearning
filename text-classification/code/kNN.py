
# coding: utf-8

# In[43]:

#Import everything
import csv,math,nltk,operator,os,pickle,re,random,scipy,sklearn,string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
sw=set(stopwords.words("english"))
from multiprocessing import Pool
from itertools import chain
import numpy as np
import os
#from tensorflow.contrib import learn
#import tensorflow as tf
#from classifierTF import *

def clean(text,deepClean):
    """
    Taken from `Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014)` and modified
    """
    text=text.strip().lower()
    text = re.sub(u'<.*?>', '', text)
    if deepClean:
        text=text.translate(str.maketrans('','',string.punctuation))
        text = re.sub(r"[^A-Za-z]", " ", text)
        text = re.sub(r"\W", " ", text)
        text = re.sub(r"  ", " ", text)
        return text
    #text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\W*\'\W*s", " \'s", text)
    text = re.sub(r"\W*\'\W*ve", " \'ve", text)
    text = re.sub(r"\W*n\W*\'\W*t", " n\'t", text)
    text = re.sub(r"\W*\'\W*re", " \'re", text)
    text = re.sub(r"\W*\'\W*d", " \'d", text)
    text = re.sub(r"\W*\'\W*ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def split(X,Y,ratio=0.9): # Split into training and validation sets
    n=len(X)
    trainIndices=random.sample(range(0,n),int(n*ratio))
    testIndices=list(set(range(n))-set(trainIndices))
    # Split by train/test
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    for i in trainIndices:
        X_train.append(X[i])
        Y_train.append(Y[i])
        
    for i in testIndices:
        X_test.append(X[i])
        Y_test.append(Y[i])

    assert len(trainIndices)+len(testIndices)==n
    return [X_train,X_test,Y_train,Y_test]

def sync_shuffle(a, b): # syncronize shuffling of x and y
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    l1 = []
    l2 = []
    for i in p:
        l1.append(a[i])
        l2.append(b[i])
    return l1, l2

def getAccuracy(y_test, y_predicted):
    return sklearn.metrics.confusion_matrix(y_test, y_predicted)

def predict(X_train, Y_train, X_predict, k=5):
    predictions = []
    for X_predict_row in X_predict:
        predictions.append(decideClass(getNeighbours(X_train, Y_train, X_predict_row,k)))
    return predictions
    
def decideClass(neighbours):
    # decide the class of a point based on the class of it's neighbours
    # majority vote wins
    # neighbours - 2d array - list of points with class value in last pos.
    votes = {}
    for i in range(len(neighbours)):
        vote = neighbours[i]
        if vote in votes:
            votes[vote] += 1
        else:
            votes[vote] = 1
    votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return votes[0][0]

def getNeighbours(X_train, Y_train, X_predict_row, k):
    # computes distance between each training point
    # and the point we try to predict
    # X_train - 2d array - m x n
    # X_predict_row - 1d array - m x 1
    # k - no. of neighbours to return
    distances = []
    for i in range(0,X_train.shape[0]):
        distance = getDistance(X_train[i],X_predict_row)
        distances.append((Y_train[i], distance))
    distances.sort(key=operator.itemgetter(1))
    
    # return list of k nearest points
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours

def getDistance(n1, n2):
    # returns distance between two arbitray data points
    # n1, n2 1d arrays - two points in dataset
    return sklearn.metrics.pairwise.euclidean_distances(n1,n2)[0][0]


# In[ ]:




# In[2]:

# if os.path.isfile("data\sparse_deep_clean_data.pickle"):
#     maxLength, phrases, testPhrases, labels = pickle.load(open("data\sparse_deep_clean_data.pickle", 'rb'))

if False:
    print(Nothing)
    
else: 
    phrases=[]
    STEMM=True # Put this to True
    REMOVESTOPWORDS=True # Put this to True for better non-Tensorflow results
    maxLength=0
    with open("data/train_input.csv","r") as f:
        reader=csv.reader(f,delimiter=",",quotechar='"')
        for i,row in enumerate(reader):
            if i>0:
                text=row[1]
                text=clean(text,deepClean=REMOVESTOPWORDS)
                if REMOVESTOPWORDS:
                    text=word_tokenize(text)
                    text=[snowball_stemmer.stem(word) if STEMM else word for word in text if word not in sw]
                    text=" ".join(text)
                maxLength=max(maxLength,len(text.split(" ")))
                phrases.append(text)

    testPhrases=[]
    with open("data/test_input.csv","r") as f:
        reader=csv.reader(f,delimiter=",",quotechar='"')
        for i,row in enumerate(reader):
            if i>0:
                text=row[1]
                text=clean(text,deepClean=REMOVESTOPWORDS)
                if REMOVESTOPWORDS:
                    text=word_tokenize(text)
                    text=[snowball_stemmer.stem(word) if STEMM else word for word in text if word not in sw]
                    text=" ".join(text)
                maxLength=max(maxLength,len(text.split(" ")))
                testPhrases.append(text)

    labels=[]
    with open("data/train_output.csv","r") as f:
        reader=csv.reader(f,delimiter=",",quotechar='"')
        for i,row in enumerate(reader):
            if i>0:
                labels.append(row[1])
    labels_factors={}
    count=0
    for label in labels:
        if label not in labels_factors.keys():
            labels_factors[label]=count
            count+=1
    for i in range(0,len(labels)):
        labels[i]= labels_factors[labels[i]]

    pickle.dump([maxLength, phrases, testPhrases, labels],open("data\sparse_deep_clean_data.pickle", 'wb'))

assert(len(labels)==len(phrases))

phrases, labels = sync_shuffle(phrases, labels)
X_train,X_test,Y_train,Y_test = split(phrases, labels)


# In[3]:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.0009, stop_words='english', smooth_idf=True, norm="l2", sublinear_tf=False, use_idf=True)


# In[4]:

X_train_tfidf = vectorizer.fit_transform(X_train)
X_new_tfidf = vectorizer.transform(X_test)


# In[5]:

X = X_train_tfidf.toarray()


# In[6]:

X = np.hstack((X,np.array(Y_train)[:, None]))


# In[ ]:

preds = []
limit = X_new_tfidf.shape[0]
for x in range(limit):
    print(x)
    predicted = predict(X_train_tfidf,Y_train,X_new_tfidf[x])
    preds.append(predicted)


# In[62]:

print(preds)
confusion = getAccuracy(Y_test[0:10],preds[0:10])


# In[67]:

print(confusion)

