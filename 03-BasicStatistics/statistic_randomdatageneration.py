# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs

conf = SparkConf().setAppName('Random Data Generation').setMaster('local[2]')
sc = SparkContext(conf=conf)

# 服从N(0, 1)的标准正太分布
u = RandomRDDs.normalRDD(sc, 1000000L, 10)
print(u.take(10))

# 转换成服从N(1, 4)的标准正太分布
v = u.map(lambda x : 1.0 + 2.0 * x)
print(v.take(10))
sc.stop()