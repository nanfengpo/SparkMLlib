# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Binary Classification Evaluation').setMaster('local[2]')
sc = SparkContext(conf=conf)



sc.stop()