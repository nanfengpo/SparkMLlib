# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Binary Classification Evaluation').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile('../data/sample_svm_data.txt')
parseData = data.map(parsePoint)

training, test = parseData.randomSplit([0.6, 0.4], seed=11L)
training.cache()

model = LogisticRegressionWithLBFGS.train(training)
predictionAndLabels = test.map(lambda lp : (float(model.predict(lp.features)), lp.label))

# Instantiate metric object
metric = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print('Area under PR = %s' % metric.areaUnderPR)

# Area under ROC curve
print('Area under ROC = %s' % metric.areaUnderROC)

sc.stop()