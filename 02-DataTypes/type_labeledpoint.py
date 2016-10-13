# coding=utf-8

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors, SparseVector, Matrix, Matrices

pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0,2], [1.0, 3.0]))

print('pos')
print(pos.label)
print(pos.features.toArray())
print('neg')
print(neg.label)
print(neg.features.toArray())