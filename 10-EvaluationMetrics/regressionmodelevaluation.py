# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.linalg import DenseVector

conf = SparkConf().setAppName('Regression model Evaluation').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data
def parsePoint(line):
    values = line.split()
    return LabeledPoint(float(values[0]), DenseVector([float(x.split(':')[1]) for x in values[1:]]))

data = sc.textFile('../data/sample_linear_regression_data.txt')
parseData = data.map(parsePoint)

# build the model
model = LinearRegressionWithSGD.train(parseData)

# get predictions
valuesAndPreds = parseData.map(lambda p : (float(model.predict(p.features)), p.label))

# instantiate metric object
metric = RegressionMetrics(valuesAndPreds)

# squared error
print('MSE=' + str(metric.meanSquaredError))

# r-squared
print('r-squared='+ str(metric.r2))

# mean absolute
print('mae=' + str(metric.meanAbsoluteError))

# explained variance
print('explained variance=' + str(metric.explainedVariance))


sc.stop()