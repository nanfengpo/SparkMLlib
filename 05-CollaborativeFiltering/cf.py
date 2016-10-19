# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

conf = SparkConf().setAppName('Collaborative Filtering').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load and parse the data
data = sc.textFile('../data/als.data')
ratings = data.map(lambda l : l.split(',')).map(lambda r : Rating(int(r[0]), int(r[1]), float(r[2])))

# build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# evaluate the model on training data
testData = ratings.map(lambda p : (p[0], p[1]))
predictions = model.predictAll(testData).map(lambda r : ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r : ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r : (r[1][0]-r[1][1])**2).mean()
print('mean square error : ' + str(MSE))

# save and load model
model.save(sc, '../model/myCollaborativeFilter')
sameModel = MatrixFactorizationModel.load(sc, '../model/myCollaborativeFilter')


sc.stop()