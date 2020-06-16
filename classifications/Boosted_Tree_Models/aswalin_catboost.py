# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:12:26 2020

@author: EmreKARA
"""

import catboost as cb
from catboost.utils import get_roc_curve
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold


def select_model(x, y, model, params,  cat_features=None):
    grid_search = GridSearchCV(model, params, scoring ='roc_auc_ovr_weighted', cv = 5, n_jobs=-1)
    grid_search.fit(x, y, cat_features = cat_features)
    print('Best Estimator:', grid_search.best_estimator_,'\n'+'Best Score:',grid_search.best_score_)

def plot_feature_importance(model, data):
    feature_importance_coef_all = model.get_feature_importance().tolist()
    feature_importance_coef_selected_index = sorted(range(len(feature_importance_coef_all)),key=feature_importance_coef_all.__getitem__, reverse=False)[:10]
    feature_names = [data.columns[i] for i in feature_importance_coef_selected_index]
    feature_importance_degrees = [feature_importance_coef_all[i] for i in feature_importance_coef_selected_index]
    feature_importance = [feature_names, feature_importance_degrees]
    
    plt.rcdefaults()
    fig = plt.figure('Feature Importance Degree From CatBoost', figsize=(15,8))
    ax = fig.add_subplot(1,1,1)
    
    y_pos = np.arange(len(feature_importance[0]))
    ax.barh(y_pos, feature_importance[1], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_importance[0])
    ax.invert_yaxis()
    ax.set_xlabel('importance degree')
    ax.set_title('Most Important 20 Features')
    
    for i, v in enumerate(feature_importance[1]):
        ax.text(v, i, " {:.2f}".format(v), color='orange', va='center', fontweight='bold')
    plt.show()
    plt.savefig('catboost_feature_importance_2.png')

def plot_roc_curves(target):
    pass
    


data= pd.read_csv("variants_not_encoded.csv")
x = data.drop(['rcv.clinical_significance'], axis=1).to_numpy()
y = data['rcv.clinical_significance'].to_numpy()


#With Categorical features
cat_features_index = [data.columns.get_loc(col) for col in data if data[col].dtypes.name in ['object','bool']] #Categoric Feature'ların indexini tutan dizi.
cat_features_index = cat_features_index[:-1] #Target sınıf çıkarılır


cbClassifier = cb.CatBoostClassifier()
param_dist = {'depth': [4, 7, 10],
          'learning_rate' : [0.01, 0.1, 0.2],
         'l2_leaf_reg': [1,4,9],
         'iterations': [200],
         'bagging_temperature':[0.3 , 1, 10],
         'grow_policy':['SymmetricTree', 'Depthwise', 'Lossguide']}
select_model(x, y, model=cbClassifier, params=param_dist, cat_features= cat_features_index)

#
#x_train, x_test, y_train, y_test = train_test_split(data.drop(['rcv.clinical_significance'], axis=1), data['rcv.clinical_significance'],random_state=10, test_size=0.2)
#cb_classifier = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=31, depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15, silent=True)
#cb_classifier.fit(x_train,y_train, cat_features= cat_features_index)
#y_pred = cb_classifier.predict(x_test)
#y_pred = y_pred.tolist()
#plot_feature_importance(cb_classifier,data)
#
#cb_classifier = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=31, depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15, silent=True)

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
#    cb_classifier.fit(x_train,y_train, cat_features= cat_features_index)
#    #plot_feature_importance(clf,data)
#    
#    y_pred = cb_classifier.predict(x_test)
#    y_pred = y_pred.tolist()
#    print('part:'+str(counter),metrics.classification_report(y_test, y_pred))
#    counter += 1
##    plot_roc_curves(data['rcv.clinical_significance'])
##    catboost_pool = cb.Pool(x_train,y_train, cat_features= cat_features_index)
##    (fpr, tpr, thresholds) = get_roc_curve(cb_classifier, catboost_pool, plot=True)
#    
##    unique, counts = np.unique(y_train, return_counts=True)
##    print('y_train', np.asarray((unique, counts)))
##    
##    unique, counts = np.unique(y_test, return_counts=True)
##    print('y_test', np.asarray((unique, counts)))







    


