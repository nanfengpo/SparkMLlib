# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.stat import KernelDensity

conf = SparkConf().setAppName('Kernel density estimation').setMaster('local[2]')
sc = SparkContext(conf=conf)
data = sc.parallelize([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0])

kd = KernelDensity()
kd.setSample(data)
data_kd = kd.setBandwidth(3.0)
densities = kd.estimate([-1.0, 2.0, 5.0])
print(densities)

sc.stop()