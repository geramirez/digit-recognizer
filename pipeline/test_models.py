import csv
import logging
import pandas as pd
import numpy as np
import random

from scipy.stats import zscore

from sklearn.metrics import accuracy_score  # , precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC


def separate_labels(data):
    """ Splits the label columns from the data """
    # Fetch a pixel column mask
    pixel_col_msk = data.columns.str.contains(
        'pixel|Solidity|AspectRatio|Perimeter|Area|Angle'
    )
    # Get the label column
    label_column = data.loc[:,data.columns.str.contains('digit|label')]
    # Get the pixel columns
    data = data.loc[:, pixel_col_msk]
    return label_column, data


def open_data(file_location):
    """ Open data using pandas, write into pandas dataframe """
    return pd.read_csv(file_location)


def predict_number(training_data, validation_data, cols):
    """ Predict the output for a specific number """
    training_labels, training_data  = separate_labels(training_data)
    clf = SVC(probability=True)
    clf.fit(training_data.loc[:, cols], training_labels)
    return clf.predict(validation_data.loc[:, cols])

def get_number_predictions():
    """ Collect predictions for numbers """
    predictions = pd.DataFrame()
    validation_labels, validation_data = separate_labels(open_data('../newdata/train_FS.csv'))
    predictions['labels'] = validation_labels.label
    with open('columns_test.csv', 'r') as csvfile:
        cols_data = csv.reader(csvfile)
        for idx, cols in enumerate(cols_data):
            data = open_data(
                '../newdata/Train_RS_scaled/trainRS-digit' + str(idx) + '.csv')
            predictions[str(idx)] = predict_number(data, validation_data, cols)
        predictions.to_csv('results.csv', index=False)


if __name__ == '__main__':
    get_number_predictions()
