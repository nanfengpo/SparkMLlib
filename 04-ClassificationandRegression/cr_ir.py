# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint, IsotonicRegression, IsotonicRegressionModel
from pyspark.mllib.util import MLUtils
import math

conf = SparkConf().setAppName('Isotonic Regression').setMaster('local[2]')
sc =SparkContext(conf=conf)

# load and parse data
def parsePoint(labeledData):
    return (labeledData.label, labeledData.features[0], 1.0)

data = MLUtils.loadLibSVMFile(sc, '../data/sample_isotonic_regression_libsvm_data.txt')
# crate label, feature, weight tuples from input data with weight set to default value 1.0
parsedData = data.map(parsePoint)

# split the data into training and test sets
(trainingData, testData) = parsedData.randomSplit([0.7, 0.3])

# create isotonic regression model from training data
model = IsotonicRegression.train(trainingData)
print('model:')
print(model)
# create tuples of predicted and real labels
predictionAndLabel = testData.map(lambda p : (model.predict(p[1]), p[0]))

# calculate mean squared error between predicted and real labels
meanSquaredError = predictionAndLabel.map(lambda pl : math.pow((pl[0]-pl[1]), 2)).mean()
print('mean squared error :' + str(meanSquaredError))

# save and load model
model.save(sc, '../model/myIsotonicRegressionModel')
sameModel = IsotonicRegressionModel.load(sc, '../model/myIsotonicRegressionModel')

sc.stop()