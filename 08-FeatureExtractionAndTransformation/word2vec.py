# coding=utf-8

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec

conf = SparkConf().setAppName('Word2Vec').setMaster('local[2]')
sc = SparkContext(conf=conf)

inp = sc.textFile('../data/sample_lda_data.txt').map(lambda row : row.split(' '))

model = Word2Vec().fit(inp)

synonyms = model.findSynonyms('1', 5)
for word, cosine_distance in synonyms:
    print('{}:{}'.format(word, cosine_distance))

sc.stop()