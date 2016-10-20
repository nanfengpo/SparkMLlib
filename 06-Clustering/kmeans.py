# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
import numpy as np
import math

conf = SparkConf().setAppName('KMeans').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse data
data = sc.textFile('../data/kmeans_data.txt')
parseData = data.map(lambda line : np.array([float(x) for x in line.split(' ')]))

# build the model
clusters = KMeans.train(parseData, 2, maxIterations=10, runs=10, initializationMode='random')

#evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return math.sqrt(sum([x**2 for x in (point-center)]))

WSSSE = parseData.map(lambda p : error(p)).reduce(lambda x, y : x+y)
print('Within Set Sum of Squared Error :' + str(WSSSE))

# save and load model
clusters.save(sc, '../model/KMeansModel')
sameModel = KMeansModel.load(sc, '../model/KMeansModel')

sc.stop()