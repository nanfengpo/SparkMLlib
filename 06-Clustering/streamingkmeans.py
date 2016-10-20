# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import numpy as np

conf = SparkConf().setAppName('Streaming KMeans').setMaster('local[2]')
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 1)

# we make an input stream of vectors for training, as well as a stream of vector for testing
def parse(lp):
    label = float(lp[lp.find('(') + 1: lp.find(')')])
    vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))
    return LabeledPoint(label, vec)

trainingData = sc.textFile("../data/kmeans_data.txt")\
    .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))

testingData = sc.textFile("../data/streaming_kmeans_data_test.txt").map(parse)

trainingQueue = [trainingData]
testingQueue = [testingData]

trainingStream = ssc.queueStream(trainingQueue)
testingStream = ssc.queueStream(testingQueue)

# we create a mdoel with random cluster and specify the number of clusters to find
model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)

# start train
model.trainOn(trainingStream)

result = model.predictOnValues(testingStream.map(lambda lp : (lp.label, lp.features)))
result.pprint()

ssc.start()
ssc.stop(stopSparkContext=True, stopGraceFully=True)


sc.stop()