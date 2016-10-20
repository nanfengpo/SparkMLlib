# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import BisectingKMeans, BisectingKMeansModel
import numpy as np

conf = SparkConf().setAppName('Bisecting KMeans').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
data = sc.textFile('../data/kmeans_data.txt')
parseData = data.map(lambda line : np.array([float(x) for x in line.split(' ')]))

# build the model
model = BisectingKMeans.train(parseData, 2, maxIterations=5)

# evaluate clustering
cost = model.computeCost(parseData)
print('bisecting K-means:', cost)

# save and load model
model.save(sc, '../model/BisectingKMeansModel')
sameModel = BisectingKMeansModel.load(sc, '../model/BisectingKMeansModel')

sc.stop()