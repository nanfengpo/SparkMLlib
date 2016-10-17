# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Decision Tree Regression').setMaster('local[2]')
sc =SparkContext(conf=conf)

# load data
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')
# split the data into training and test sets
(training, testData) = data.randomSplit([0.7, 0.3])

# training a decision tree regression
model = DecisionTree().trainRegressor(training, categoricalFeaturesInfo={}, impurity='variance', maxDepth=5, maxBins=32)

# evaluate model on test instance and compute test error
predictions = model.predict(testData.map(lambda x : x.features))
labelAndPredictions = testData.map(lambda x : x.label).zip(predictions)
testMSE = labelAndPredictions.map(lambda (v, p) : (v - p)**2).sum()/float(testData.count())

print('test mean squared error :' + str(testMSE))
print('learned regression tree model :')
print(model.toDebugString())

# save and load model
model.save(sc, '../model/myDecisionTreeRegressionModel')
sameModel = DecisionTreeModel.load(sc, '../model/myDecisionTreeRegressionModel')


sc.stop()