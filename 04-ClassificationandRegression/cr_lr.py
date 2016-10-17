# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName('Logistic Regression').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile('../data/sample_svm_data.txt')
parseData = data.map(parsePoint)

# build the model
model = LogisticRegressionWithLBFGS.train(parseData)

# evaluating the model on training data
labelsAndPreds = parseData.map(lambda p : (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p) : v != p).count()/float(parseData.count())
print('training error :' + str(trainErr))

# save and load model
model.save(sc, '../model/pythonLogisticRegressionWithLBFGSModel')
sameModel = LogisticRegressionModel.load(sc, '../model/pythonLogisticRegressionWithLBFGSModel')

sc.stop()