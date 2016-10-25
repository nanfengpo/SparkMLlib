# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Multiclass Classification Evaluation').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load training data
data = MLUtils.loadLibSVMFile(sc, '../data/sample_multiclass_classification_data.txt')

# split data into training and test
training, test = data.randomSplit([0.6, 0.4], seed=11)
training.cache()

# run training algorithm to build the model
model = LogisticRegressionWithLBFGS.train(training, numClasses=3)

# compute raw scores on the test set
predictionAndLabels = test.map(lambda lp : (float(model.predict(lp.features)), lp.label))

# instanticate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

print('summary stats')
print('precision= %s' % metrics.precision())
print('recall= %s' % metrics.recall())
print('F1 Score= %s' % metrics.fMeasure())
sc.stop()