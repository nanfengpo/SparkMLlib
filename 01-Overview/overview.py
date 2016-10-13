# coding=utf-8

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0,2], [1.0, 3.0]))

print(pos)
print(neg)