# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Gradient Boosted Tree Regression').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data file
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')

# split the data into training and test
trainingData, test = data.randomSplit([0.7, 0.3])

# train a gradient boosted tree model
model = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo={}, numIterations=3)

# evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x : x.features))
labelsAndPredictions = test.map(lambda lp : lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p) : (v-p)**2).sum()/float(test.count())
print('test mean squared error :' + str(testMSE))
print('learned regression GBT model :')
print(model.toDebugString)

# save and load
model.save(sc, '../model/myGradientBoostingRegressionModel')
sameModel = GradientBoostedTreesModel.load(sc, '../model/myGradientBoostingRegressionModel')

sc.stop()