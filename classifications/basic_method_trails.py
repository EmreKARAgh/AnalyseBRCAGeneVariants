# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import preprocessing

def unique(list1):       
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    return unique_list
pp = preprocessing.Preprocess(data_path_c = 'variants_encoded.csv')
x_train, x_test, y_train, y_test = pp.trainTestSplitting(['rcv.clinical_significance'])

log_r = LogisticRegression()
log_r.fit(x_train, y_train.values.ravel())
y_pred_log = log_r.predict(x_test)
y_pred_log = list(y_pred_log)
cm_log = confusion_matrix(y_test,y_pred_log)
print('cm_log:\n',cm_log)

knn = KNeighborsClassifier()
knn.fit(x_train,y_train.values.ravel())
y_pred_knn = knn.predict(x_test)
y_pred_knn = list(y_pred_knn)
cm_knn=confusion_matrix(y_test,y_pred_knn)
print('cm_knn:\n',cm_knn)

all_classes = unique(y_test.iloc[:,0])
all_classes_log = unique(y_pred_log)
all_classes_knn = unique(y_pred_knn)
print('u:',len(all_classes) , all_classes)
print('u_log:',len(all_classes_log) , all_classes_log)
print('u_knn:',len(all_classes_knn) , all_classes_knn)


#svc = SVC(kernel='linear')
#svc.fit(x_train,y_train)
#y_pred_svc = svc.predict(x_test)
#cm_svc = confusion_matrix(y_test,y_pred_svc)
#print('cm_svc:',cm_svc)

