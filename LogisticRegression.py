#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:27:26 2018

@author: nehansh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:11:01 2018

@author: cs
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random as rn
import sklearn as sk
import pdb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

class ProcessData:
    
    def readintoDF(self, Filename):
        df = pd.read_csv(Filename, sep = '\t')
        return df
        
    def readyData(self, patientL, path):
        filePathList = []
        label = []
        for index, row in patientL.iterrows():
            patient = row['SubjectID']
            filepath = path + str(patient) + ".txt"
            label.append(row['label'])
            filePathList.append(filepath)
            
        listCUIs = []
        for filePath in filePathList:
            with open (filePath) as fin:
                for line in fin:
                    listCUIs.append(line)
        data_df = pd.DataFrame({'PatientCUIs': listCUIs, 'label': label})
        return data_df, listCUIs, label


    def processText(self, listCUIs, label):
        
        X_train, X_test, y_train, y_test = train_test_split(listCUIs, label, test_size = 0.15, random_state=42)

        return X_train, y_train, X_test, y_test
    
    
     
    def getInitialVariables(self, listCUIs):
        CUIsLength = []
        for x in listCUIs:
            y = x.split()
            z = len(y)
            #print(z)
            CUIsLength.append(z)
        highest = max(CUIsLength)
        #print(highest)
        vocab_size = len(listCUIs)
        return highest, vocab_size
      
     
    
if __name__ == "__main__":
    
    PD = ProcessData()
    patientL = PD.readintoDF("patientLabel.txt")
#    patientL = patientL.head(150)
    #print(patientL)
    
    fullData_df, listCUIs, label = PD.readyData(patientL, "ClusterDataCUI/")
    
#    Maxlen, vocabSize = PD.getInitialVariables(listCUIs)
#    print("vocabSize: " + str(vocabSize))
#    print("Maxlen:" + str(Maxlen))

    X_train, y_train, X_test, y_test = PD.processText(listCUIs, label)
#    print("a: " + str(vocab_size))
    
    #Pipeline for fiting the algorithm & vectorizer on trainining data
pipe_svc = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', linear_model.LogisticRegression(random_state=1)),
])

#Grid Search Implementation
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range}]


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)


#fitting grid search after k-fold split
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# fitting the best estimator on train set
clf = gs.best_estimator_
clf.fit(X_train, y_train)
# predictedClf = clf.predict(X_test)
#Calculating accuracy on Validation Data Set
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))
# print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=predictedClf))
# print(metrics.classification_report(y_test, predictedClf))
# print(metrics.confusion_matrix(y_test, predictedClf))

#Working on Test Data

predictedClf = clf.predict(X_test)

# # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# # print(confmat)

print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=predictedClf))
print(metrics.classification_report(y_test, predictedClf))
print(metrics.confusion_matrix(y_test, predictedClf))
    

    
    
    