# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Random Forest Classification').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data file
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')

# split the data into training and test
trainingData, test = data.randomSplit([0.7, 0.3])

# train a random forest model
# note: use large numTree in practice
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy='auto', impurity='gini',
                                     maxDepth=4, maxBins=32)

# evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda  x : x.features))
labelsAndPredictions = test.map(lambda lp : lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p) : v != p).count()/float(test.count())

print('test error :' + str(testErr))
print('learned classification forest model :')
print(model.toDebugString)

# save and load model
model.save(sc, '../model/myRandomForestClassificationModel')
sameModel = RandomForestModel.load(sc, '../model/myRandomForestClassificationModel')

sc.stop()