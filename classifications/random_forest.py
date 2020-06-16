# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 01:22:24 2020

@author: EmreKARA
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

def select_model(x, y, model, params):
    grid_search = GridSearchCV(model, params, scoring ='roc_auc_ovr_weighted', cv = 5, n_jobs=-1)
    grid_search.fit(x, y)
    print('Best Estimator:', grid_search.best_estimator_,'\n'+'Best Score:',grid_search.best_score_)
def all_classification_reports(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)

data = pd.read_csv('variants_encoded.csv')
x = data.drop(["rcv.clinical_significance"], axis=1)
y = data["rcv.clinical_significance"]
le = LabelEncoder()
y = le.fit_transform(y)

sc = StandardScaler()
x = sc.fit_transform(x)

param_dist = {'bootstrap': [True, False],
 'max_depth': [10, 30, 50, 70, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 1000, 1400, 1800],
 'criterion': ['gini','entropy']
 }
param_dist = {'bootstrap': [False],
 'max_depth': [70],
 'max_features': ['auto'],
 'min_samples_leaf': [4],
 'min_samples_split': [5],
 'n_estimators': [200],
 'criterion': ['gini']
 }
#select_model(x, y, model=rfc, params=param_dist)

rfc = RandomForestClassifier(bootstrap=False, max_depth=70, max_features='auto', min_samples_leaf=4, min_samples_split=5, n_estimators=200, criterion='gini')

x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=10, test_size=0.2)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
y_pred = y_pred.tolist()
cm = metrics.classification_report(y_test,y_pred)
print(cm)
#
#
#
#originalclass = []
#predictedclass = []
#target_names = ['Benign', 'Likely beging', 'Likely pathogenic', 'Pathegenic']
#skf = StratifiedKFold(n_splits=5)
#counter = 0
#for train_index, test_index in skf.split(x,y):
#    x_train = x[train_index]
#    x_test = x[test_index] 
#    
#    y_train = y[train_index]
#    y_test = y[test_index]    
#    
#    #select_model(x_train, y_train)
#    logr.fit(x_train,y_train)
#    #plot_feature_importance(clf,data)
#    
#    y_pred = logr.predict(x_test)
#    y_pred = y_pred.tolist()
#    counter += 1
#    
#    all_classification_reports(y_test, y_pred)
#
#print(metrics.classification_report(originalclass, predictedclass, target_names = target_names))


