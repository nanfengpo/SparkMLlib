# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, StreamingLinearRegressionWithSGD
import sys

conf = SparkConf().setAppName('Streaming linear regression').setMaster('local[2]')
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 1)

# load data
def parse(lp):
    label = float(lp[lp.find('(')+1 : lp.find(',')])
    vec = Vectors.dense(lp[lp.find('[')+1 : lp.find(']')]).split(',')
    return LabeledPoint(label, vec)

trainingData = ssc.textFileStream(sys.argv[1]).map(parse).cache()
testData = ssc.textFileStream(sys.argv[1]).map(parse)

# build the model
numFeatures = 3
model = StreamingLinearRegressionWithSGD()
model.setInitialWeights([0.0, 0.0, 0.0])
model.trainOn(trainingData)
print(model.predictOnValues(testData.map(lambda lp : (lp.lable, lp.features))))
ssc.start()
ssc.awaitTermination()
#ssc.stop()
sc.stop()