# coding=utf-8

# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Matrices, Matrix, Vectors
import numpy as np

## 假设检验
conf = SparkConf().setAppName('Hypothesis testing').setMaster('local[2]')
sc = SparkContext(conf=conf)
vec = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25)
goodnessOfFitTestResult = Statistics.chiSqTest(vec)
print(goodnessOfFitTestResult)

mat = Matrices.dense(3, 2, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
independenceTestResult = Statistics.chiSqTest(mat)
print(independenceTestResult)

parallelData = sc.parallelize([0.1, 0.15, 0.2, 0.3, 0.25])
testResult = Statistics.kolmogorovSmirnovTest(parallelData, 'norm', 0, 1)
print(testResult)
sc.stop()