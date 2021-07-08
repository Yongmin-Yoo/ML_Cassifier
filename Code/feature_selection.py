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


def coefficient_Logistic(X_train,y_train):
    model_lr = LogisticRegression()
    model_lr.fit(X_train,y_train)
    model_coef = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['Model Coefficient'])
    model_coef.sort_values(by='Model Coefficient', ascending=False, inplace=True)
    plt.bar(model_coef.index, model_coef['Model Coefficient'])
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()
    
    
def coefficient_XGB(X_train,y_train):
    model_xgb = XGBClassifier()
    model_xgb.fit(X_train,y_train)
    fig = plt.figure(figsize=(10,10))
    plt.barh(X.columns, model_xgb.feature_importances_)
    
    
    
def coefficient_SVM(X_train,y_train):
    model_svm = SVC(kernel='linear')
    model_svm.fit(X_train,y_train)
    model_coef = pd.DataFrame(data=model_svm.coef_[0], index=X.columns, columns=['Model Coefficient'])
    model_coef.sort_values(by='Model Coefficient', ascending=False, inplace=True)
    plt.bar(model_coef.index, model_coef['Model Coefficient'])
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()
    
    

def coefficient_Randomforest(X_train,y_train):
    model_RDF = RandomForestClassifier()
    model_RDF.fit(X_train,y_train)
    fig = plt.figure(figsize=(10,10))
    plt.barh(X.columns, model_RDF.feature_importances_)

    
    
# def coefficient_NaiveBayes(X_train,y_train):
#     model_lr = GaussianNB()
#     model_lr.fit(X_train,y_train)
#     model_coef = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['Model Coefficient'])
#     model_coef.sort_values(by='Model Coefficient', ascending=False, inplace=True)
#     plt.bar(model_coef.index, model_coef['Model Coefficient'])
#     plt.xticks(rotation=90)
#     plt.grid()
#     plt.show()
    
    
def coefficient_Gradientboosting(X_train,y_train):
    model_GB = GradientBoostingClassifier()
    model_GB.fit(X_train,y_train)
    fig = plt.figure(figsize=(10,10))
    plt.barh(X.columns, model_GB.feature_importances_)


    
# def coefficient_KNN(X_train,y_train):
#     model_lr = KNeighborsClassifier()
#     model_lr.fit(X_train,y_train)
#     model_coef = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['Model Coefficient'])
#     model_coef.sort_values(by='Model Coefficient', ascending=False, inplace=True)
#     plt.bar(model_coef.index, model_coef['Model Coefficient'])
#     plt.xticks(rotation=90)
#     plt.grid()
#     plt.show()
    
    
def coefficient_DecisionTree(X_train,y_train):
    model_tree = DecisionTreeClassifier()
    model_tree.fit(X_train,y_train)
    fig = plt.figure(figsize=(10,10))
    plt.barh(X.columns, tree.feature_importances_)
