# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.stat import Statistics
import numpy as np

## 汇总统计
conf = SparkConf().setAppName('Summary Statistics').setMaster('local[2]')
sc = SparkContext(conf=conf)

mat = sc.parallelize([np.array([1.0, 10.0, 100.0]),
                      np.array([2.0, 20.0, 200.0]),
                      np.array([3.0, 30.0, 300.0])])
summary = Statistics.colStats(mat)

# 统计出列的各种信息
print(summary.mean())        # 每列的均值
print(summary.variance())    # 每列的方差
print(summary.numNonzeros()) # 每列的非零值的个数

sc.stop()