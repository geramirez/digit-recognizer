import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

print('Running...')

traindata = np.loadtxt('c:\\users\\samarths\\desktop\\the62s\\newtrain620.csv', delimiter = ',', skiprows = 1)

nums, attribs1 = np.hsplit(traindata, [1])
binarynums, attribs = np.hsplit(attribs1, [1])

images = attribs.view()

labels = binarynums.ravel().astype(np.int)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images)
data = images.reshape((n_samples, -1))

print(labels.shape)
print(data.shape)
print(n_samples)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the training set
classifier.fit(data, labels)

testdata = np.loadtxt('c:\\users\\samarths\\desktop\\the62s\\newtest620.csv', delimiter = ',', skiprows = 1)

testnums, testattribs1 = np.hsplit(testdata, [1])
testbinarynums, testattribs = np.hsplit(testattribs1, [1])

testimages = testattribs.view()

expected = testbinarynums.ravel().astype(np.int)

n_testsamples = len(testimages)
testdatareshaped = testimages.reshape((n_testsamples, -1))

print(expected.shape)
print(testdatareshaped.shape)
print(n_testsamples)

predicted = classifier.predict(testdatareshaped)

print(predicted.shape)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
