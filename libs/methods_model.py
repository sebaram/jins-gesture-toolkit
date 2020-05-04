# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:21:48 2020

@author: JY
"""
from joblib import dump

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


    
class linearSVMclassifier:
    def __init__(kernel="linear", C=0.025):
        self.clf = SVC(kernel=kernel, C=C)
    def train(X,y):
        self.clf.fit(X, y)
    def get_confusion_matrix(self, testX, testy, cv=10):
        cross_result = cross_val_score(self.clf, testX, testy, cv=10)
        y_pred = cross_val_predict(self.clf, testX, testy, cv=10)
        conf_mat = confusion_matrix(testy, y_pred)
        
        return conf_mat

class rbfSVMclassifier:
    def __init__(gamma=2, C=1):
        self.clf = SVC(gamma=gamma, C=C)
    def train(X,y):
        self.clf.fit(X, y)
    def get_confusion_matrix(self, testX, testy, cv=10):
        cross_result = cross_val_score(self.clf, testX, testy, cv=10)
        y_pred = cross_val_predict(self.clf, testX, testy, cv=10)
        conf_mat = confusion_matrix(testy, y_pred)
        
        return conf_mat
    
class MLPclassifier:
    def __init__(alpha=1, max_iter=1000):
        self.clf = MLPClassifier(alpha=alpha, max_iter=max_iter)
    def train(X,y):
        self.clf.fit(X, y)
    def get_confusion_matrix(self, testX, testy, cv=10):
        cross_result = cross_val_score(self.clf, testX, testy, cv=10)
        y_pred = cross_val_predict(self.clf, testX, testy, cv=10)
        conf_mat = confusion_matrix(testy, y_pred)
        
        return conf_mat
    

class RDFclassifier:
    def __init__(N_ESTIMATORS=50):
        self.clf = RandomForestClassifier(n_estimators = N_ESTIMATORS)
    def train(X,y):
        self.clf.fit(X, y)
    def get_confusion_matrix(self, testX, testy, cv=10):
        cross_result = cross_val_score(self.clf, testX, testy, cv=10)
        y_pred = cross_val_predict(self.clf, testX, testy, cv=10)
        conf_mat = confusion_matrix(testy, y_pred)
        
        return conf_mat
    
    
class AdaBoostclassifier:
    def __init__():
        self.clf = AdaBoostClassifier()
    def train(X,y):
        self.clf.fit(X, y)
    def get_confusion_matrix(self, testX, testy, cv=10):
        cross_result = cross_val_score(self.clf, testX, testy, cv=10)
        y_pred = cross_val_predict(self.clf, testX, testy, cv=10)
        conf_mat = confusion_matrix(testy, y_pred)
        
        return conf_mat