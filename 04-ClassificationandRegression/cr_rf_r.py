# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Random Forest Regression').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data file
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')

# split the data into training and test
trainingData, test = data.randomSplit([0.7, 0.3])

# train a random forest model
# note: use larger numTrees in practice
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    numTrees=3, featureSubsetStrategy='auto',
                                    impurity='variance', maxDepth=4, maxBins=32)

# evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x : x.features))
labelAndPredictions = test.map(lambda lp : lp.label).zip(predictions)
testMSE = labelAndPredictions.map(lambda (v, p) : (v-p)**2).sum()/float(test.count())
print('test mean squared error :' + str(testMSE))
print('learned regression forest model :')
print(model.toDebugString)

# save and load model
model.save(sc, '../model/myRandomForestRegressionModel')
sameModel = RandomForestModel.load(sc, '../model/myRandomForestRegressionModel')

sc.stop()