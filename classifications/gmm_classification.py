import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

import pandas as pd


data = pd.read_csv('variants_continous_encoded.csv')
#data = pd.read_csv('variants_continous_encoded_only_target.csv') spherical train: 15.263954997836434 spherical test: 14.993523316062177 diag train: 15.253137170056252 diag test: 14.993523316062177 tied train: 36.7914322803981 tied test: 36.36658031088083 full train: 0.8437905668541757 full test: 0.8743523316062176
target = data.loc[:,data.columns =='rcv.clinical_significance'].to_numpy()
data = data.loc[:, data.columns != 'rcv.clinical_significance'].to_numpy()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(data, target)))
#
#
X_train = data[train_index]
y_train = target[train_index].reshape(-1)
X_test = data[test_index]
y_test = target[test_index].reshape(-1)
xtr = X_train
xte = X_test
#
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
