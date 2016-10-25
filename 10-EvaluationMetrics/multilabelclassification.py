# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Multilabel Classification Evaluation').setMaster('local[2]')
sc = SparkContext(conf=conf)

scoreAndLabels = sc.parallelize([ # 此数据有误
    ([0.0, 1.0], [0.0, 2.0]),
    ([0.0, 2.0], [0.0, 1.0]),
    ([], [0.0]),
    ([2.0], [2.0]),
    ([2.0, 0.0], [2.0, 0.0]),
    ([0.0, 1.0, 2.0], [0.0, 1.0]),
    ([1.0], [1.0, 2.0])])

# instantiate metrics object
metrics = MulticlassMetrics(scoreAndLabels)

# summary stats
print('recall:', metrics.recall())
print('precision:', metrics.precision())
print('F1 measure:', metrics.fMeasure())
print('accuracy:', metrics.accuracy())

# individual label stats

sc.stop()