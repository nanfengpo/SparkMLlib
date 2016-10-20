# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
import numpy as np

conf = SparkConf().setAppName('Gaussian Mixture').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
data = sc.textFile('../data/gmm_data.txt')
parsedData = data.map(lambda line : np.array([float(x) for x in line.strip().split(' ')]))

# build the model
gmm = GaussianMixture.train(parsedData, 2)

# save and load model
gmm.save(sc, '../model/GaussianMixtureModel')
sameModel = GaussianMixtureModel.load(sc, '../model/GaussianMixtureModel')

# output parameters of model
for i in range(2):
    print('weight=',sameModel.weights[i],
          'mu=',sameModel.gaussians[i].mu,
          'sigma=',sameModel.gaussians[i].sigma.toArray())

sc.stop()