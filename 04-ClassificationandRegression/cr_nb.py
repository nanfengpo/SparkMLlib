# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Naive Bayes').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data file
data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')

# split data approximately into training and test
training, test = data.randomSplit([0.6, 0.4])

# train a naive bayes model
model = NaiveBayes.train(training, 1.0)

# make prediction and test accuracy
predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
accuracy = 1.0*predictionAndLabel.filter(lambda (v, p) : v==p).count()/test.count()
print('model accuracy :' + format(accuracy))

# save and load model
output_dir = '../model/myNaiveBayesModel'
# MLUtils.rmtree(output_dir, ignore_errors=True)
model.save(sc, output_dir)
sameModel = NaiveBayesModel.load(sc, output_dir)
predictionAndLabel = test.map(lambda p : (sameModel.predict(p.features), p.label))
accuracy = 1.0*predictionAndLabel.filter(lambda (v, p) : v==p).count()/test.count()
print('sameModel accuracy :' + format(accuracy))

sc.stop()