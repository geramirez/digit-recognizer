#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
"""
 Digit-Recognizer dimension reduction
"""
#-------------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
import sys
import os.path
#
import numpy as np
print('Numpy         version:', np.__version__)
#
import matplotlib
print('Matplotlib    version:', matplotlib.__version__)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#
import pandas as pd
print('Pandas        version:', pd.__version__)
#
import sklearn
print('sklearn       version:', sklearn.__version__)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
from sklearn import metrics

#
#-------------------------------------------------------------------------------
#

INPUT_DATA_PATH=os.path.expanduser('~/Desktop/DataScience/UW-DataScience-450/captstone/datasets')
OUTPUT_DATA_PATH=os.path.expanduser('~/Desktop/DataScience/UW-DataScience-450/capstone/datasets')
#
TEST_DATA=os.path.join(OUTPUT_DATA_PATH, 'test_FS.csv')
TRAINING_DATA=os.path.join(OUTPUT_DATA_PATH, 'train_FS.csv')

#
#-------------------------------------------------------------------------------
#

def train_model(label, X_train, y_train, cv_folds=0):
    #

    NUM_TREES = 30
    VERBOSITY=0
    NUM_JOBS=10

    print('\n',60 * '-')
    print('Modelling %s\n' % label)
    clf = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=VERBOSITY,
                    random_state=42,
                    n_jobs=NUM_JOBS)
    if cv_folds:
        scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
        print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
        print(scores)

    clf.fit(X_train, y_train)
    prediction_pdata = clf.predict_proba(X_train)
    prediction = clf.predict(X_train)


    model_metrics = metrics.classification_report(y_train, prediction, [1, 0])
    print(model_metrics)
    print('Confusion Matrix\n', metrics.confusion_matrix(y_train, prediction))

    #  save model
    filename = 'classifiers/clf-%s.pkl' % (label)
    joblib.dump(clf, filename)

    return clf, prediction, prediction_pdata
#
#-------------------------------------------------------------------------------
#

def main():
    """
    """
    print('Loading training data')
    training_data = pd.read_csv(TRAINING_DATA)
    print('Samples: %d, attributes: %d' %(training_data.shape[0],
        training_data.shape[1]))

    print('Loading test data')
    test_data = pd.read_csv(TEST_DATA)
    print('Samples: %d, attributes: %d' %(test_data.shape[0],
        test_data.shape[1]))


    y = training_data['label']
    X = training_data.drop('label', axis=1)

    #
    #  Binarize target value
    #
    y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
    y_bin = np.column_stack((y, y_bin))

    labels = ['label', 'digit0','digit1','digit2','digit3','digit4',
                       'digit5','digit6','digit7','digit8','digit9']

    y_bin = pd.DataFrame(y_bin, columns=labels)

    #
    #  Reduce dimensionality, using PCA
    #
    print('Selecting Features')
    pca = PCA(n_components=0.95, whiten=True)

    # combine both training and test data for PCA
    XX = X.append(test_data)
    pca.fit(XX)
    print('N-Features selected:', len(pca.explained_variance_ratio_))

    print('Applying feature selection to training and test data')
    Xp = pca.transform(X)
    Zp = pca.transform(test_data)

    #
    #    Train Digit Classifier Models
    #

    #
    #  Digit0 Model
    #
    prediction = np.array(y, np.int)
    probability = np.array(y, np.float)

    #
    #
    for digit in labels[1:]:
        clf, pred, pdata = train_model(digit, Xp, y_bin[digit], cv_folds=0)

        digit_pred = np.array(pred, np.int)
        prediction = np.column_stack((prediction, digit_pred))

        digit_prob = np.array(pdata[:,1], np.float)
        probability = np.column_stack((probability, digit_prob))

        for i in range(len(y_bin[digit])):
            if y_bin[digit].iloc[i] != pred[i]:
                print('mismatch at %d, y = %d, prediction = %d' % (i, y_bin[digit].iloc[i], pred[i]) )

    print('predictions = ', prediction.shape)
    df = pd.DataFrame(prediction, columns=labels)
    PREDICTIONS=os.path.join(OUTPUT_DATA_PATH, 'Digit_Prediction.csv')
    df.to_csv(PREDICTIONS, index=False)


    PROBABILITES=os.path.join(OUTPUT_DATA_PATH, 'Digit_p.csv')
    print('probabilites = ', probability.shape)
    df = pd.DataFrame(probability, columns=labels)
    df.to_csv(PROBABILITES, index=False)


    y = df['label']
    X = df.drop('label', axis=1)

    NUM_TREES = 10
    VERBOSITY=0
    NUM_JOBS=10

    print('\n',60 * '-')
    clf = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=VERBOSITY,
                    random_state=42,
                    n_jobs=NUM_JOBS)
    scores = cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    print(scores)

    clf.fit(X, y)
    prediction = clf.predict(X)

    model_metrics = metrics.classification_report(y, prediction, [0,1,2,3,4,5,6,7,8,9])
    print(model_metrics)
    print('Confusion Matrix\n', metrics.confusion_matrix(y, prediction))



#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
