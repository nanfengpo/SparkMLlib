# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth

conf = SparkConf().setAppName('FP-Growth').setMaster('local[2]')
sc = SparkContext(conf=conf)

data = sc.textFile('../data/sample_fpgrowth.txt')
transactions = data.map(lambda line : line.strip().split(' '))

model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
result = model.freqItemsets().collect()

print(result)

sc.stop()