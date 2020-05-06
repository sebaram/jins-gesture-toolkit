# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:21:48 2020

@author: JY
"""
import sys
from joblib import dump, load
from datetime import datetime


from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class Classifier:
    """Super Class"""
    def __init__(self):
        self.target_names_list = ""
        self.save_name = "not saved..."
        
    def train(self, Xlist,ylist, save=False, target_names_list=""):
        self.target_names_list = target_names_list
        self.clf.fit(Xlist, ylist)
        if save:
            self.save_model()
    
    def get_confusion_matrix(self, testX, testy, cv=10, target_names_list=""):
        self.target_names_list = target_names_list
        cross_result = cross_val_score(self.clf, testX, testy, cv=cv)
        y_pred = cross_val_predict(self.clf, testX, testy, cv=cv)
        conf_mat = confusion_matrix(testy, y_pred)
        
        return cross_result, conf_mat
    
    def save_model(self, f_name=""):
        if f_name =="":
            f_name = datetime.now().strftime('%Y-%m-%d %H_%M_%S')+"_{}.joblib".format(self.__class__.__name__)
        self.save_name = f_name
        dump(self, f_name)
        
        
    
class linearSVMclassifier(Classifier):
    def __init__(self, kernel="linear", C=0.025):
        super().__init__() 
        self.clf = SVC(kernel=kernel, C=C)

class rbfSVMclassifier(Classifier):
    def __init__(self, gamma=2, C=1):
        super().__init__() 
        self.clf = SVC(gamma=gamma, C=C)
    
class MLPclassifier(Classifier):
    def __init__(self, alpha=1, max_iter=1000):
        super().__init__() 
        self.clf = MLPClassifier(alpha=alpha, max_iter=max_iter)
    

class RDFclassifier(Classifier):
    def __init__(self, N_ESTIMATORS=50):
        super().__init__() 
        self.clf = RandomForestClassifier(n_estimators = N_ESTIMATORS)
    
    
class AdaBoostclassifier(Classifier):
    def __init__(self):
        super().__init__() 
        self.clf = AdaBoostClassifier()


def load_model(f_name):
    return load(f_name)
    
if __name__ == "__main__":
    aa = linearSVMclassifier()
    aa.save_model('test_lin_self.joblib')
    aa.save_model()
    #%%
    load_dict = load('test_lin.joblib')
    load_dict2 = load('test_lin_self.joblib')
    
