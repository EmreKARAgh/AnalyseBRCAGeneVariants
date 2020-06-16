import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import preprocessing

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

import pandas as pd


#data = pd.read_csv('variants_encoded.csv') # indexli hali daha yüksek doğruluk veriyor???
data = pd.read_csv('variants_encoded_NoIndex.csv') # indexli hali daha yüksek doğruluk veriyor???
##data = pd.read_csv('variants_continous_encoded_only_target.csv')


#pp=preprocessing.Preprocess('variants_encoded_NoIndex.csv') #Scaling'de başarımı oldukça düşürdü
#data = pp.getData()
#columns_to_be_scaled = list(data.columns)
#columns_to_be_scaled.remove('rcv.clinical_significance')
#pp.scale(columns_to_be_scaled)
#data = pp.getData()



target = data.loc[:,data.columns =='rcv.clinical_significance'].to_numpy()
data = data.loc[:, data.columns != 'rcv.clinical_significance'].to_numpy()


# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(data, target)))


X_train = data[train_index]
y_train = target[train_index].reshape(-1)
X_test = data[test_index]
y_test = target[test_index].reshape(-1)
xtr = X_train
xte = X_test

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {cov_type: GaussianMixture(n_components=n_classes,
              covariance_type=cov_type, max_iter=20, random_state=0)
              for cov_type in ['spherical', 'diag', 'tied', 'full']}

n_estimators = len(estimators)

for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.

#    print(np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)]))
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)



    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print(name,'train:',train_accuracy)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print(name,'test:',test_accuracy)
