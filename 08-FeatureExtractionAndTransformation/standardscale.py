# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Standard Scaler').setMaster('local[2]')
sc = SparkContext(conf=conf)

data = MLUtils.loadLibSVMFile(sc, '../data/sample_libsvm_data.txt')
label = data.map(lambda x : x.label)
features = data.map(lambda x : x.features)

scaler1 = StandardScaler().fit(features)
scaler2 = StandardScaler(withMean=True, withStd=True).fit(features)

data1 = label.zip(scaler1.transform(features))
data2 = label.zip(scaler2.transform(features.map(lambda x : Vectors.dense(x.toArray()))))

print(data1.first())
print(data2.first())

sc.stop()