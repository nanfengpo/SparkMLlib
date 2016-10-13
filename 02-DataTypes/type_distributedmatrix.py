# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors, SparseVector, Matrix, Matrices
from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRow, IndexedRowMatrix, CoordinateMatrix, MatrixEntry, BlockMatrix
from pyspark.sql import SQLContext

conf = SparkConf().setAppName('Data Types Index Row Matrix').setMaster('local[2]')
sc = SparkContext(conf=conf)

## 分布式矩阵

# 行矩阵
rowMatrix = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# Create a RowMatrix from an RDD of vectors.
mat = RowMatrix(rowMatrix)
# Get its size.
m = mat.numRows()  # 4
n = mat.numCols()  # 3
# Get the rows as an RDD of vectors again.
rowsRDD = mat.rows
print('行矩阵:')
print(mat)
print(m, n)
print(rowsRDD.collect())

# 行索引矩阵
# Create an RDD of indexed rows.
#   - This can be done explicitly with the IndexedRow class:
indexedRows = sc.parallelize([IndexedRow(0, [1, 2, 3]),
                              IndexedRow(1, [4, 5, 6]),
                              IndexedRow(2, [7, 8, 9]),
                              IndexedRow(3, [10, 11, 12])])
#   - or by using (long, vector) tuples:
# indexedRows = sc.parallelize([(0, [1, 2, 3]), (1, [4, 5, 6]),
#                               (2, [7, 8, 9]), (3, [10, 11, 12])])
# solve the question:AttributeError: 'PipelinedRDD' object has no attribute 'toDF'
sqlContext = SQLContext(sc)
# Create an IndexedRowMatrix from an RDD of IndexedRows.
mat = IndexedRowMatrix(indexedRows)
# Get its size.
m = mat.numRows()  # 4
n = mat.numCols()  # 3
# Get the rows as an RDD of IndexedRows.
rowsRDD = mat.rows
# Convert to a RowMatrix by dropping the row indices.
rowMat = mat.toRowMatrix()
print('行索引矩阵:')
print(m, n)
print(rowMat)

# 三元组分布式矩阵
# Create an RDD of coordinate entries.
#   - This can be done explicitly with the MatrixEntry class:
entries = sc.parallelize([MatrixEntry(0, 0, 1.2), MatrixEntry(1, 0, 2.1), MatrixEntry(6, 1, 3.7)])
#   - or using (long, long, float) tuples:
entries = sc.parallelize([(0, 0, 1.2), (1, 0, 2.1), (2, 1, 3.7)])
# Create an CoordinateMatrix from an RDD of MatrixEntries.
mat = CoordinateMatrix(entries)
# Get its size.
m = mat.numRows()  # 3
n = mat.numCols()  # 2
# Get the entries as an RDD of MatrixEntries.
entriesRDD = mat.entries
# Convert to a RowMatrix.
rowMat = mat.toRowMatrix()
# Convert to an IndexedRowMatrix.
indexedRowMat = mat.toIndexedRowMatrix()
# Convert to a BlockMatrix.
blockMat = mat.toBlockMatrix()
print('三元组分布式矩阵:')
print(blockMat)

# 块矩阵
# Create an RDD of sub-matrix blocks.
blocks = sc.parallelize([((0, 0), Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])),
                         ((1, 0), Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12]))])

# Create a BlockMatrix from an RDD of sub-matrix blocks.
mat = BlockMatrix(blocks, 3, 2)
# Get its size.
m = mat.numRows() # 6
n = mat.numCols() # 2
# Get the blocks as an RDD of sub-matrix blocks.
blocksRDD = mat.blocks
# Convert to a LocalMatrix.
localMat = mat.toLocalMatrix()
# Convert to an IndexedRowMatrix.
indexedRowMat = mat.toIndexedRowMatrix()
# Convert to a CoordinateMatrix.
coordinateMat = mat.toCoordinateMatrix()
print('块矩阵:')
print(coordinateMat)
sc.stop()