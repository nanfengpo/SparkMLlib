# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF, IDF

conf = SparkConf().setAppName('TF-IDF').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load documents
documents = sc.textFile('../data/kmeans_data.txt').map(lambda line : line.split(' '))
hashingTF = HashingTF()
tf = hashingTF.transform(documents)

tf.cache()

idf = IDF().fit(tf)
tfidf = idf.transform(tf)

idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idf.transform(tf)


sc.stop()