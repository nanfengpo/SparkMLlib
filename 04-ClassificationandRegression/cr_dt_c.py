# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Decision Tree Classification').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data file into an RDD of LabelPoint
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')

#split the data into training and test sets
(trainingData, testData) = data.randomSplit([0.7, 0.3])

#train a decision tree model
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)

# evaluate model on test instance and compute test error
predictions = model.predict(testData.map(lambda x : x.features))
labelAndPredictions = testData.map(lambda lp : lp.label).zip(predictions)
testErr = labelAndPredictions.filter(lambda (v, p) : v != p).count()/float(testData.count())

print('test err' + str(testErr))
print('learned classification tree model :' + str(model.toDebugString))

# save and load model
model.save(sc, '../model/myDecisionTreeClassificationModel')
sameModel = DecisionTreeModel.load(sc, '../model/myDecisionTreeClassificationModel')

sc.stop()