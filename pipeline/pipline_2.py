import csv
import logging
import pandas as pd
import numpy as np
import random

from scipy.stats import zscore

from sklearn.metrics import accuracy_score  # , precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC


def open_data(file_location):
    """ Open data using pandas, write into pandas dataframe """
    return pd.read_csv(file_location)


def separate_labels(data):
    """ Splits the label columns from the data """
    # Fetch a pixel column mask
    pixel_col_msk = data.columns.str.contains(
        'pixel|Solidity|AspectRatio|Perimeter|Area|Angle'
    )
    # Get the label column
    label_column = data.loc[:,data.columns.str.contains('digit')]
    # Get the pixel columns
    data = data.loc[:, pixel_col_msk]
    return label_column, data


def prep_data(data):
    """ This function contains a few functions for preping the data
    these functions will be removed later """
    labels, data = separate_labels(data)
    data = data.apply(zscore, axis=0)
    # Reinsert the labels
    data['label'] = labels
    # Drop the empty columns
    data = data.dropna(axis=1)
    return data


def split_data(data):
    """ Split data into a test, training, and validation set
    using a 60, 20, 20 ratio """
    # Create a numpy array of randoms
    randarray = np.random.rand(len(data))
    # Create a training, test mask, and validation mask
    train_msk = (randarray >= .2) & (randarray <= .8)
    test_msk = randarray < .2
    validate_msk = randarray > .8
    # Apply the masks and return data
    training_data = data[train_msk]
    testing_data = data[test_msk]
    validation_data = data[validate_msk]
    return training_data, testing_data, validation_data


def extract_features(training_data, testing_data, classifier,
                     number_of_cols=10, number_of_models=3):
    """ Extract features by building multiple models and scoring features
    that appear in models that have an accuracy score above the threshold """
    results = []
    # Extract labels from data frames
    training_data_label, training_data = separate_labels(training_data)
    testing_data_label, testing_data = separate_labels(testing_data)
    # Start iteration
    for idx in range(number_of_models):
        # Get the colums to be fit
        cols = random.sample(range(training_data.shape[1]), number_of_cols)
        # Initalize classifier
        clf = classifier()
        # Fit the model
        clf.fit(training_data.iloc[:, cols], training_data_label)
        # Get predictions
        predictions = clf.predict(testing_data.iloc[:, cols])
        # Calculate accuracy
        accuracy = accuracy_score(testing_data_label, predictions)
        # Append results
        results.append([accuracy] + cols)

    # Clean up mess
    training_data['label'] = training_data_label
    testing_data['label'] = testing_data_label
    return results


def process_results(results):
    """ Covert results into a list of important variables"""
    # Use the results to extract important columns
    clf_results = pd.DataFrame(results)
    important_cols = []
    mean_result = clf_results.iloc[:, 0].mean()
    for row in clf_results.iterrows():
        if row[1][0] > mean_result:
            important_cols.extend(row[1][1:])
    return pd.Series(important_cols).value_counts()


def get_optimize_result(training_data, validation_data, important_cols_result):
    """ Get the number of cols that gets the best score """
    last_score = 0.0
    new_score = 0.0
    number_of_cols = 1
    decreases = 0
    optimal_result = {'score': 0.0, 'number_of_cols': 1}
    # Extract labels from data frames
    training_data_label, training_data = separate_labels(training_data)
    validation_data_label, validation_data = separate_labels(validation_data)
    while True:
        cols = important_cols_result.index[0: number_of_cols]
        # Fit models and test
        clf = SVC()
        clf.fit(training_data.iloc[:, cols], training_data_label)
        predictions = clf.predict(validation_data.iloc[:, cols])
        new_score = accuracy_score(validation_data_label, predictions)
        if new_score < optimal_result['score']:
            optimal_result['score'] = new_score
            optimal_result['number_of_cols'] = number_of_cols
        if last_score > new_score:
            decreases += 1
            if decreases > 3:
                break
        last_score = new_score
        number_of_cols += 5
    cols = important_cols_result.index[0: number_of_cols]
    export_optimal_result(training_data.iloc[:, cols].columns)


def export_optimal_result(data):
    with open('columns_individuals.csv', 'a') as csvfile:
        result_writer = csv.writer(csvfile)
        result_writer.writerow(data)


def digit_train(data_folder):
    """ Loop through digit data picking columns for each digit """
    for i in range(10):
        data = open_data(data_folder + str(i) + '.csv')
        logging.info('Data Loaded for #%s', i)
        training_data, testing_data, validation_data = split_data(data)
        logging.info('Data Split for #%s', i)
        results = extract_features(
            training_data=training_data,
            testing_data=testing_data,
            classifier=DTC,
            number_of_cols=20,
            number_of_models=500
        )
        logging.info('Data extracted')
        important_columns = process_results(results)
        logging.info('Data import columns')
        get_optimize_result(training_data, validation_data, important_columns, i)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    digit_train('../newdata/Train_RS_scaled/trainRS-digit')
    """

    data = open_data('../data/train.csv')
    logging.info('Data Loaded')
    data = prep_data(data)
    logging.info('Data Preped')
    training_data, testing_data, validation_data = split_data(data)
    logging.info('Data Split')
    results = extract_features(
        training_data=training_data, testing_data=testing_data, classifier=DTC,
        number_of_cols=20, number_of_models=1000)
    logging.info('Data extracted')
    important_columns = process_results(results)
    logging.info('Data import columns')
    get_optimize_result(training_data, validation_data, important_columns)
    """
