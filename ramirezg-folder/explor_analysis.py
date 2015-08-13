import logging

import pandas as pd
import numpy as np
from scipy.stats import zscore
import scipy.cluster.hierarchy as sch

""" Script attempting do run some hierarchical clustering, which failed
because the there are too many attributes """

def load_data():
    """ Load data using pandas """
    data = pd.read_csv('data/train.csv')
    return data

def randomize_data(data):
    """ Randomize data by label """
    # Create new column to store random values
    data['random'] = 0.0
    ## Iterate through data to fill random column
    for label in data.label.unique():
        shape = data.loc[data.label == label, 'random'].shape
        data.loc[data.label == label, 'random'] = np.random.rand(shape[0])
    return data

def normalize_data(data):
    """ Normalize data """
    ## Get only columns that have pixel in the title
    pixel_cols = data.columns.str.contains('pixel')
    ## Save columns that don't
    label_columns = data.iloc[:,[1,-1]]
    ## Put the df back together while normalizing the other columns
    ## This method saves memory
    data = data.loc[:, pixel_cols].apply(zscore, axis=0)
    data['labels'] = label_columns.iloc[:,0]
    data['random'] = label_columns.iloc[:,1]
    return data

def hierarchical_clust(data):
    """ See http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage """
    # use a small segment of the data
    test_data = data[data.random > .8]
    result = sch.linkage(test_data.iloc[:,:-2], method='single')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    data = load_data()
    logging.info('Data Loaded')
    data = randomize_data(data)
    logging.info('Data Randomization applied')
    data = normalize_data(data)
    logging.info('Data normalized')
    # Drop columns with nas, mostly drops the side columns
    data = data.dropna(axis=1)
    logging.info('NaNs Dropped')
    # hierarchical_clust(data)
    logging.info('Clust finished')
