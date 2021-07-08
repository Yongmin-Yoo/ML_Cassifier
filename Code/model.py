import os
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score , KFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def Logistic_classifier(X_train,y_train,y_test):
    
    model_lr = LogisticRegression()
    model_lr.fit(X_train,y_train)
    pred = model_lr.predict(X_test)
    
    return print(classification_report(y_test,pred))



def XGB_classifier(X_train,y_train,y_test):
    
    model_xgb = XGBClassifier()
    model_xgb.fit(X_train,y_train)
    pred = model_xgb.predict(X_test)

    return print(classification_report(y_test,pred))


def SVM_classifier(X_train,y_train,y_test):
    
    model_svm = SVC()
    model_svm.fit(X_train,y_train)
    pred = model_svm.predict(X_test)

    print(classification_report(y_test,pred))
    
    
def RandomForest_classifier(X_train,y_train,y_test):
    
    model_RDF = RandomForestClassifier()
    model_RDF.fit(X_train,y_train)
    pred = model_RDF.predict(X_test)

    print(classification_report(y_test,pred))
    

    
def NaiveBayes_classifier(X_train,y_train,y_test):
    
    model_NB = GaussianNB()
    model_NB.fit(X_train,y_train)
    pred = model_NB.predict(X_test)

    print(classification_report(y_test,pred))
    
    
    
def GradientBoosting_classifier(X_train,y_train,y_test):
    model_GBC = GradientBoostingClassifier()
    model_GBC.fit(X_train,y_train)
    pred = model_GBC.predict(X_test)

    return print(classification_report(y_test,pred))



def knn_classifier(X_train,y_train,y_test):
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train,y_train)
    pred = model_knn.predict(X_test)

    print(classification_report(y_test,pred))

    
def decision_tree(X_train,y_train,y_test):
    model_tree = DecisionTreeClassifier()
    model_tree.fit(X_train,y_train)
    pred = model_tree.predict(X_test)

    print(classification_report(y_test,pred))


def all(X_train,y_train,y_test):
    print("-----------------------------------------------------------")
    print("------------------- Logistic Regression -------------------")
    print("-----------------------------------------------------------")
    print(Logistic_classifier(X_train,y_train,y_test))
    print("-----------------------------------------------------------")
    print("-------------------    XGB_classifier   -------------------")
    print("-----------------------------------------------------------")    
    print(XGB_classifier(X_train,y_train,y_test))
    print("-----------------------------------------------------------")
    print("-------------------   SVM_classifier    -------------------")
    print("-----------------------------------------------------------")
    print(SVM_classifier(X_train,y_train,y_test))
    print("-----------------------------------------------------------")
    print("----------------- RandomForest_classifier -----------------")
    print("-----------------------------------------------------------")
    print(RandomForest_classifier(X_train,y_train,y_test))
    print("-----------------------------------------------------------")
    print("------------------ NaiveBayes_classifier ------------------")
    print("-----------------------------------------------------------")
    print(NaiveBayes_classifier(X_train,y_train,y_test))
    print("-----------------------------------------------------------")
    print("---------------  GradientBoostingClassifier ---------------")
    print("-----------------------------------------------------------")
    print(GradientBoosting_Classifier(X_train,y_train,y_test))    
    print("-----------------------------------------------------------")
    print("----------------------  knn_classifier---------------------")
    print("-----------------------------------------------------------")
    print(knn_classifier(X_train,y_train,y_test))
    print("-----------------------------------------------------------")
    print("-------------------decision_classifier---------------------")
    print("-----------------------------------------------------------")
    print(decision_tree(X_train,y_train,y_test))
