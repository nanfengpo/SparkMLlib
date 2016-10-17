# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName('Linear Support Vector Machines').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile('../data/sample_svm_data.txt')
parseData = data.map(parsePoint)

# build the model
model = SVMWithSGD.train(parseData, iterations=100)

# evaluating the model on training data
labelsAndPreds = parseData.map(lambda p : (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p) : v != p).count()/float(parseData.count())

print('training error :' + str(trainErr))

# save and load model
model.save(sc, '../model/pythonSVMWithSGDModel')
sameModel = SVMModel.load(sc, '../model/pythonSVMWithSGDModel')

sc.stop()