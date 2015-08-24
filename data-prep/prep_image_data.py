#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
"""
 Digit-Recognizer image data pre-processing
 - data cleanup, normalization, etc.
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
import cv2
print('OpenCV        version:', cv2.__version__)
import time
from collections import namedtuple

#-------------------------------------------------------------------------------

#
#
INPUT_DATA_PATH=os.path.expanduser('~/Desktop/DataScience/UW-DataScience-450/kaggle_comp')
OUTPUT_DATA_PATH=os.path.expanduser('~/Desktop/DataScience/UW-DataScience-450/capstone/data-prep')

#
#
TRAINING_FILE=os.path.join(INPUT_DATA_PATH, 'train.csv')
TEST_FILE=os.path.join(INPUT_DATA_PATH, 'test.csv')


OUTPUT_DATA_FILE=os.path.join(OUTPUT_DATA_PATH, 'train_FS.csv')

#-------------------------------------------------------------------------------

SampleDigit = namedtuple('SampleDigit', ['image','label'], verbose=False)
IMAGE=0
LABEL=1

#-------------------------------------------------------------------------------

max_height = 0
max_width = 0


def plot_image(image, title=''):
    """
    """
    plt.title(title)
    plt.matshow(image)

def threshold_image(image, threshold=170):
    """
    """
    image = image.astype('uint8')

    thresh_image = cv2.adaptiveThreshold(image, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,7,0)

    return thresh_image

def to_top_left(sample_image):
    """
    """
    global max_width, max_height

    image = sample_image.astype('uint8')

    contours = exterior_contour(image)

    cnt = contours[0][0]
    x,y,w,h = cv2.boundingRect(cnt)

    if w > max_width: max_width = w
    if h > max_height: max_height = h

    rows,cols = image.shape

    M = np.float32([[1,0,-x],[0,1,-y]])

    translated_image = cv2.warpAffine(image ,M, (cols,rows))

    return translated_image

def crop_image(image, xRange, yRange):
    """
    """
    cropped = image[xRange[0]:xRange[1], yRange[0]:yRange[1]]
    return cropped

def exterior_contour(sample_image):
    """
    """
    image = sample_image.copy()
    image = image.astype('uint8')

    contours = cv2.findContours(image,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)

    return contours


def resize_image(image, scale=0.6):
    """
    """
    image = image.astype('uint8')
    resized_image = cv2.resize(image, None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    return resized_image

def image_metrics(sample_image):
    """
    """
    image = sample_image.copy()
    image = image.astype('uint8')

    contours = cv2.findContours(image,
                                cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_NONE)
    count = len(contours)

    cnt = contours[0][0]
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area >= 0.001:
        solidity = float(area)/hull_area
    else:
        solidity = 0.0
    #print('Solidity 0.3%f' % (solidity))

    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    #print('Aspect ratio: %0.3f' % (aspect_ratio))

    perimeter = cv2.arcLength(cnt,True)
    string = 'Perimeter = %0.3f' % (perimeter)
    #print(string)

    area = cv2.contourArea(cnt)
    #print('Area = 0.3%f' % (area))

    feature_dict = {'Solidity' : solidity,
                    'AspectRatio' : aspect_ratio,
                    'Perimeter' : perimeter,
                    'Area' : area,
                    'Contours' : cnt }

    return feature_dict


#-------------------------------------------------------------------------------

class DigitData(object):
    """
    """
    def __init__(self, csv_data_filename=TRAINING_FILE, df=None):

        if csv_data_filename:
            self.source_data_filename = os.path.expanduser(csv_data_filename)
            self.df = pd.read_csv(self.source_data_filename)
            print('Loaded samples: %d, attributes: %d' %
                    (self.df.shape[0], self.df.shape[1]))
        else:
            self.df = df

    def extra_features(self, index):
        """
        """
        s = self.df.iloc[index,:]

        if len(s) == 785:
            return []
        else:
            return s[1:5]

    def instance(self, index):
        """
        Returns SampleDigit(image, label) for instance at index
        """
        s = self.df.iloc[index,:]

        if 'label' in self.df.columns:
            # Training data includes a label
            label = s[0]
            if len(s) == 785:
                idx = 1
            else:
                idx = 5
            image = s[idx:].values.reshape(28,28)
        else:
            # test data
            label = None
            image = s.values.reshape(28,28)

        return SampleDigit(image, label)

    def plot_instance(self, index):
        """
        """
        sample = self.instance(index)

        title = ' '
        if sample[LABEL]:
            title = sample[LABEL]

        plot_image(sample[IMAGE], title)


#-------------------------------------------------------------------------------


def main():
    """
    """
    print('Load training data')
    training_data = DigitData(TRAINING_FILE)

    #
    #   Translate images to upper left corner
    #

    d = None
    instance_count, attribute_count = training_data.df.shape
    for id in range(instance_count):

        sample = training_data.instance(id)

        mdict = image_metrics(sample[IMAGE])
        metrics = [mdict['Solidity'],
                   mdict['AspectRatio'],
                   mdict['Perimeter'],
                   mdict['Area']]

        image = to_top_left(sample[IMAGE])
        image = image.astype('uint8')
        attributes = np.append(sample[LABEL], metrics)
        extended_attributes = np.append(attributes, image)

        if  d:
            d.extend(list(extended_attributes))
        else:
            d = list(extended_attributes)

        print('Processing instance: %d' % (id))

    #
    array = np.array(d)
    num_attributes = 785 + len(metrics)
    column_names = list(training_data.df.columns)
    metrics_names = ['Solidity', 'AspectRatio', 'Perimeter', 'Area']
    nlist = [column_names[0]]
    nlist.extend(metrics_names)
    nlist.extend(column_names[1:])
    column_names = nlist
    new_data = array.reshape(array.size/num_attributes, num_attributes)
    df = pd.DataFrame(data=new_data, columns=column_names)
    data = DigitData(None, df)

    #---------------------------------------------------------------------------

    #
    #   Crop images to remove blank border
    #

    xBound = max_width
    yBound = max_height
    d =None
    instance_count, attribute_count = data.df.shape
    for id in range(instance_count):

        sample = data.instance(id)
        image = crop_image(sample[IMAGE], (0,xBound), (0,yBound))
        #
        #  TO REDUCE IMAGE SIZE UNCOMMENT NEXT LINEß
        #
        #image = resize_image(image, 0.6)
        image = image.astype('uint8')
        metrics = data.extra_features(id)
        attributes = np.append(sample[LABEL], metrics)
        attributes = np.append(attributes, image)

        size = len(attributes)

        if  d:
            d.extend(list(attributes))
        else:
            d = list(attributes)

        print('Processing instance: %d' % (id))
    #
    array = np.array(d)
    new_data = array.reshape(array.size/(size), size)
    df = pd.DataFrame(data=new_data, columns=data.df.columns[0:size])

    df.to_csv(OUTPUT_DATA_FILE)
    print('Samples: %d, attributes: %d' %(df.shape[0], df.shape[1]))
    print('Data saved to %s' % (OUTPUT_DATA_FILE))

if __name__ == '__main__':
    main()
