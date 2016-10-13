# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Data Types').setMaster('local[2]')
sc = SparkContext(conf=conf)

# 稀疏数据
examples = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')
print(examples.collect())
sc.stop()