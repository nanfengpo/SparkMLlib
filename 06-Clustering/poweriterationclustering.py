# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel
import numpy as np

conf = SparkConf().setAppName('Power Iteration Clustering').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
data = sc.textFile('../data/pic_data.txt')
similarities = data.map(lambda line:tuple([float(x) for x in line.split(' ')]))

# cluster the data into two classes using PowerIterationClustering
model = PowerIterationClustering.train(similarities, 2, 10)
model.assignments().foreach(lambda x : (x.id, x.cluster))

# save and load model
model.save(sc, '../model/PICModel')
sameModel = PowerIterationClusteringModel.load(sc, '../model/PICModel')

sc.stop()