# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.stat import Statistics
import numpy as np

## 分层抽样
# allows users to sample approximately ⌈fk⋅nk⌉∀k∈K⌈fk⋅nk⌉∀k∈K items,
# where fkfk is the desired fraction for key kk,
#  nknk is the number of key-value pairs for key kk,
#  and KK is the set of keys
conf = SparkConf().setAppName('Stratified sampling').setMaster('local[2]')
sc = SparkContext(conf=conf)

data = sc.parallelize([(1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')])
# 指定每个key被抽取到的比例
fractions = {1:0.1, 2:0.6, 3:0.3}

approxSample1 = data.sampleByKey(False, fractions)
print(approxSample1.collect())

sc.stop()