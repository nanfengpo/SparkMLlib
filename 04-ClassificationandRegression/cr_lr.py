# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName('Logistic Regression').setMaster('local[2]')
sc = SparkContext(conf=conf)

sc.stop()