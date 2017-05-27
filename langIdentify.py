# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:57:08 2017

@author: chen
"""

import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np


import math
def entropy(somearray):
    entropy = 0
    for i in somearray:
        entropy = entropy - i * math.log(i,2)
    return entropy

def text_filter(strlang):
    url = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.IGNORECASE)
    strlang = url.sub("", strlang)
    for ch in " \n\t\r.,:;[]()\\/\"0123456789~!`?@#$%^&*_-=-":
        strlang = strlang.replace(ch, " ")
    return strlang
    
    
def getfile(filename):
    jsfile = open(filename).readlines()
    json_data = []
    for line in jsfile:
        json_data.append(json.loads(line))
    return json_data

#transform a sentence into a frequence rate array
#eg. for sentence "Ilovemachinelearning"
#for n_gram = 2, this function will find frequency of il, lo, ov, ve, em, ma ....
#and divide them by the sum of frequency
#these variables will be attributes of our train data
def char_fre_train_data(filename):
    vectorizer = TfidfVectorizer( 
                             decode_error=u'strict', strip_accents=None, 
                             lowercase=True, preprocessor=None, 
                             tokenizer=None, analyzer=u'char', 
                             stop_words=None, 
                             ngram_range=(1, 2), max_df=1.0, 
                             min_df=1, max_features=None, 
                             vocabulary=None, binary=False,  
                             norm=u'l2', use_idf=True, 
                             smooth_idf=True, sublinear_tf=False)
    json_data = getfile(filename)
    X = []
    Y = []
    i = 0
    for line in json_data:
        X.append(text_filter(json_data[i]["text"]))
        Y.append(json_data[i]["lang"])
        i = i + 1
        print(i)
    X = vectorizer.fit_transform(X)
    return (X, Y, vectorizer)
    
#Saving n-gramed training data  
import pickle   
def maketrainsample(filename):
    traindata = char_fre_train_data(filename)
    output_pickle = open('traindata.pkl', 'wb')
    pickle.dump(traindata, output_pickle)
    output_pickle.close()
    
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def pkloutput(filename):
    pkl_file = open(filename,  'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def datatrain_LogisticRegression():
    data = pkloutput('traindata.pkl')
    X = data[0]
    Y = data[1]
    classifier = LogisticRegression()
    classifier.fit(X, Y)
    #Saving trained classifier
    output_pickle = open('datatrain_LogisticRegression.pkl', 'wb')
    pickle.dump(classifier, output_pickle)
    output_pickle.close()
    return classifier
 
#function for test accuracy   
def predict_LogisticRegression_test(filetest):
    classifier = pkloutput('datatrain_LogisticRegression.pkl')
    data = pkloutput('traindata.pkl')
    vectorizerload= data[2]
    json_data = getfile(filetest)
    X = []
    Y = []
    i = 0
    for line in json_data:
        X.append(text_filter(json_data[i]["text"]))
        Y.append(json_data[i]["lang"])
        i = i + 1
    X = vectorizerload.transform(X)
    test_predictions = classifier.predict(X)
    proba = classifier.predict_proba(X)
    fixed_prediction = []
    for i in range(len(Y)):
        #low proba and high entropy mean we do not have high confidence on these prediction, so we predict as unk
        if (max(proba[i]) < 0.5 and entropy(proba[i]) > 2):
            fixed_prediction.append(u'unk')
        else:
            fixed_prediction.append(test_predictions[i])
    print 'LogisticRegression test accuarcy: %f' % accuracy_score(Y, fixed_prediction) 
    
#function for prediction
def predict_LogisticRegression(filetest):
    classifier = pkloutput('datatrain_LogisticRegression.pkl')
    data = pkloutput('traindata.pkl')
    vectorizerload= data[2]
    json_data = getfile(filetest)
    X = []
    i = 0
    for line in json_data:
        X.append(text_filter(json_data[i]["text"]))
        i = i + 1
    number_of_instance = i
    X = vectorizerload.transform(X)
    test_predictions = classifier.predict(X)
    proba = classifier.predict_proba(X)
    fixed_prediction = []
    for i in range(number_of_instance):
        if (max(proba[i]) < 0.5 and entropy(proba[i]) > 2):
            fixed_prediction.append(u'unk')
        else:
            fixed_prediction.append(test_predictions[i])
    return fixed_prediction





def datatrain_LinearSVC():
    data = pkloutput('traindata.pkl')
    X = data[0]
    Y = data[1]
    classifier = LinearSVC()
    classifier.fit(X, Y)
    output_pickle = open('datatrain_LinearSVC.pkl', 'wb')
    pickle.dump(classifier, output_pickle)
    output_pickle.close()
    return classifier
    
#function for test accuracy     
def predict_LinearSVC_test(filetest):
    classifier = pkloutput('datatrain_LinearSVC.pkl')
    data = pkloutput('traindata.pkl')
    vectorizerload= data[2]
    json_data = getfile(filetest)
    X = []
    Y = []
    i = 0
    for line in json_data:
        X.append(text_filter(json_data[i]["text"]))
        Y.append(json_data[i]["lang"])
        i = i + 1
    X = vectorizerload.transform(X)
    
    test_predictions = classifier.predict(X)
    confidence = classifier.decision_function(X)
    fixed_prediction = []
    for i in range(len(Y)):
        if (np.var(confidence[i]) < 0.1):
             fixed_prediction.append(u'unk')
        else:
            fixed_prediction.append(test_predictions[i])
    print 'LinearSVC test accuarcy: %f' % accuracy_score(Y, fixed_prediction)
    

#function for prediction      
def predict_LinearSVC(filetest):
    classifier = pkloutput('datatrain_LinearSVC.pkl')
    data = pkloutput('traindata.pkl')
    vectorizerload= data[2]
    json_data = getfile(filetest)
    X = []
    i = 0
    for line in json_data:
        X.append(text_filter(json_data[i]["text"]))
        i = i + 1
    number_of_instance = i
    X = vectorizerload.transform(X)
    
    test_predictions = classifier.predict(X)
    confidence = classifier.decision_function(X)
    fixed_prediction = []
    for i in range(number_of_instance):
        if (np.var(confidence[i]) < 0.1):
             fixed_prediction.append(u'unk')
        else:
            fixed_prediction.append(test_predictions[i])
    return fixed_prediction
    


