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
import numpy as np
import scipy.stats as stats
print('Numpy         version:', np.__version__)
import matplotlib
print('Matplotlib    version:', matplotlib.__version__)
import matplotlib.pyplot as plt
import pandas as pd
print('Pandas        version:', pd.__version__)
import sklearn
print('sklearn       version:', sklearn.__version__)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation

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

def binarize_label(series, label):
    bin_list = []

    for i in series:
        if i == label:
            bin_list.append(1)
        else:
            bin_list.append(0)


    bin_array = np.array(bin_list, dtype='int')

    return bin_array
#
#-------------------------------------------------------------------------------
#

def main():
    """
    """
    training_data = pd.read_csv(TRAINING_DATA)
    test_data = pd.read_csv(TEST_DATA)

    y = training_data['label']
    X = training_data.drop('label', axis=1)

    #
    #  Binarize target value
    #
    digit_labels = ['digit0','digit1','digit2','digit3','digit4',
                    'digit5','digit6','digit7','digit8','digit9']
    digit0 = binarize_label(y, 0)
    digit1 = binarize_label(y, 1)
    digit2 = binarize_label(y, 2)
    digit3 = binarize_label(y, 3)
    digit4 = binarize_label(y, 4)
    digit5 = binarize_label(y, 5)
    digit6 = binarize_label(y, 6)
    digit7 = binarize_label(y, 7)
    digit8 = binarize_label(y, 8)
    digit9 = binarize_label(y, 9)

    #
    #  Reduce dimensionality, using PCA
    #
    pca = PCA(n_components=0.95, whiten=True)
    pca.fit(X, y)
    print('PCA Explanined\n', pca.explained_variance_ratio_)
    Xp = pca.transform(X)
    Zp = pca.transform(test_data)

    NUM_TREES = 60

    #
    #  Digit0 Model
    #
    print('Modelling Digit 0')
    clf0 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=1,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf0, Xp, digit0, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf0.fit(Xp, digit0)
    p0 = clf0.predict_proba(Zp)

    df = pd.DataFrame({'digit0' : p0[:,1]})
    #
    #  Digit1 Model
    #
    print('Modelling Digit 1\n')
    clf1 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=1,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf1, Xp, digit1, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf1.fit(Xp, digit1)
    p1 = clf1.predict_proba(Zp)
    df['digit1'] = p1[:,1]
    #
    #  Digit2 Model
    #
    print('Modelling Digit 2\n')
    clf2 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf2, Xp, digit2, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf2.fit(Xp, digit2)
    p2 = clf2.predict_proba(Zp)
    df['digit2'] = p2[:,1]
    #
    #  Digit3 Model
    #
    print('Modelling Digit 3\n')
    clf3 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf3, Xp, digit3, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf3.fit(Xp, digit3)
    p3 = clf3.predict_proba(Zp)
    df['digit3'] = p3[:,1]
    #
    #  Digit4 Model
    #
    print('Modelling Digit 4\n')
    clf4 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    lsscores = cross_val_score(clf4, Xp, digit4, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf4.fit(Xp, digit4)
    p4 = clf4.predict_proba(Zp)
    df['digit4'] = p4[:,1]
    #
    #  Digit5 Model
    #
    print('Modelling Digit 5\n')
    clf5 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf5, Xp, digit5, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf5.fit(Xp, digit5)
    p5 = clf5.predict_proba(Zp)
    df['digit5'] = p5[:,1]
    #
    #  Digit6 Model
    #
    print('Modelling Digit 6\n')
    clf6 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf6, Xp, digit6, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf6.fit(Xp, digit6)
    p6 = clf6.predict_proba(Zp)
    df['digit6'] = p6[:,1]
    #
    #  Digit7 Model
    #
    print('Modelling Digit 7\n')
    clf7 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf7, Xp, digit7, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf7.fit(Xp, digit7)
    p7 = clf7.predict_proba(Zp)
    df['digit7'] = p7[:,1]
    #
    #  Digit8 Model
    #
    print('Modelling Digit 8\n')
    clf8 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf8, Xp, digit8, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf8.fit(Xp, digit8)
    p8 = clf8.predict_proba(Zp)
    df['digit8'] = p8[:,1]

    #
    #  Digit9 Model
    #
    print('Modelling Digit 9\n')
    clf9 = RandomForestClassifier(n_estimators=NUM_TREES,
                    criterion='entropy',
                    max_depth=None,
                    min_samples_split=2,
                    verbose=0,
                    random_state=42,
                    n_jobs=4)
    scores = cross_val_score(clf9, Xp, digit9, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    clf9.fit(Xp, digit9)
    p9 = clf9.predict_proba(Zp)
    df['digit9'] = p9[:,1]


    PREDICTIONS=os.path.join(OUTPUT_DATA_PATH, 'Digit_Prediction_P.csv')
    df.to_csv(PREDICTIONS, index=False)


#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
