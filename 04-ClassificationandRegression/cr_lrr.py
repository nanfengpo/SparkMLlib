# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

conf = SparkConf().setAppName('Linear least squares, Lasso, and ridge regression').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile('../data/lpsa.data')
parseData = data.map(parsePoint)

# build the model
model = LinearRegressionWithSGD.train(parseData, iterations=100, step=0.0000001)

# evaluate the model on training data
valuesAndPreds = parseData.map(lambda p : (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p) : (v - p)**2).reduce(lambda a, b : a+b)/valuesAndPreds.count()
print('mean squared error :' + str(MSE))

# save and load model
model.save(sc, '../model/pythonLinearRegressionWithSGDModel')
sameModel = LinearRegressionModel.load(sc, '../model/pythonLinearRegressionWithSGDModel')
sc.stop()