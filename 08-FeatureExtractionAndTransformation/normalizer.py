# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Normalize').setMaster('local[2]')
sc = SparkContext(conf=conf)

data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')
labels = data.map(lambda x : x.label)
features = data.map(lambda x : x.features)

normalizer1 = Normalizer()
normalizer2 = Normalizer(p=float('inf'))

data1 = labels.zip(normalizer1.transform(features))
data2 = labels.zip(normalizer2.transform(features))

print(data.first())
print(data1.first())
print(data2.first())


sc.stop()