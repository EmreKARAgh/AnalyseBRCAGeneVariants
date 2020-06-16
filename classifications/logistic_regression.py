# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:10:20 2020

@author: EmreKARA
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
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

param_dist = {'solver': ['newton-cg', 'saga', 'lbfgs'],
  'C':[0.01,0.09,0.5,1,5,10],
 'class_weight': ['balanced', None],
 'max_iter': [50,100,250,500]}

#logr = LogisticRegression() #1 with best_params
#select_model(x, y, model=logr, params=param_dist)

logr = LogisticRegression(C=0.09, max_iter=50) #1 with best_params


originalclass = []
predictedclass = []
target_names = ['Benign', 'Likely beging', 'Likely pathogenic', 'Pathegenic']
skf = StratifiedKFold(n_splits=5)
counter = 0
for train_index, test_index in skf.split(x,y):
    x_train = x[train_index]
    x_test = x[test_index] 
    
    y_train = y[train_index]
    y_test = y[test_index]    
    
    #select_model(x_train, y_train)
    logr.fit(x_train,y_train)
    #plot_feature_importance(clf,data)
    
    y_pred = logr.predict(x_test)
    y_pred = y_pred.tolist()
    counter += 1
    
    all_classification_reports(y_test, y_pred)

print(metrics.classification_report(originalclass, predictedclass, target_names = target_names))
    

    


