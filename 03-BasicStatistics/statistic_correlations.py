# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.stat import Statistics
import numpy as np

## 相关性
conf = SparkConf().setAppName('Correlations').setMaster('local[2]')
sc = SparkContext(conf=conf)

# 计算列与列之间的相关性
seriesX = sc.parallelize([1.0, 10.0, 100.0])
seriesY = sc.parallelize([5.0, 30.0, 366.0])
print(Statistics.corr(seriesX, seriesY, method='pearson'))

data = sc.parallelize([np.array([1.0, 10.0, 100.0]),
                      np.array([2.0, 20.0, 200.0]),
                      np.array([5.0, 30.0, 366.0])])
print(Statistics.corr(data, method='pearson'))

sc.stop()