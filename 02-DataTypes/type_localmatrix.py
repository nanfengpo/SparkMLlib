# coding=utf-8

from pyspark.mllib.linalg import Vectors, SparseVector, Matrix, Matrices

# 本地矩阵
dm = Matrices.dense(2, 2, [2, 3, 4, 5])
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])

print('dm')
print(dm.toArray())
print('sm')
print(sm.toDense())
