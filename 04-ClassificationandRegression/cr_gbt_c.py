# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Gradient Boosted Tree Classification').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data file
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')

# split the data into training and test
trainingData, test = data.randomSplit([0.7, 0.3])

# train a gradient boost tree model
model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={},
                                             numIterations=3)

# evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x : x.features))
labelAndPredictions = test.map(lambda lp : lp.label).zip(predictions)
testErr = labelAndPredictions.filter(lambda (v, p) : v != p).count()/float(test.count())
print('test error :' + str(testErr))
print('learned classification GBT model :')
print(model.toDebugString)

# save and load
model.save(sc, '../model/myGradientBoostingClassificationModel')
sameModel = GradientBoostedTreesModel.load(sc, '../model/myGradientBoostingClassificationModel')

sc.stop()