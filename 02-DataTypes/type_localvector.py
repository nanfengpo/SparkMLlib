# coding=utf-8

from pyspark.mllib.linalg import Vectors, SparseVector, Matrix, Matrices
import numpy as np
import scipy.sparse as sps

## 本地向量
# 本地密集向量
dv1 = np.array([1.0, 0.0, 3.0])
dv2 = [1.0, 0.0, 3.0]
# 本地稀疏向量
# 参数一：元素个数
# 参数二：向量下标
# 参数三：向量值
sv1 = Vectors.sparse(3, [0,2], [1.0, 3.0])
#
sv2 = sps.csc_matrix((np.array([1.0, 3.0]), np.array([0, 2]), np.array([0, 2])), shape=(3, 1))
print(sv1.toArray())
print(sv2.toarray())