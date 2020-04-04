# to submit
# spark-submit --master spark://localhost:7077 spark-als-matrix-factorization.py

import pyspark
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import time


t0 = time.time()

def log(mssg):
    global t0
    delta = round(t0 - time.time(), 2)
    t0 = time.time()
    print(f'{delta}: {mssg}')


spark = SparkSession.builder\
    .appName('movie-recs')\
    .master('local[8]')\
    .getOrCreate()
log('Session created')


# load data
df = spark.read \
    .option("header", "true") \
    .format("csv")\
    .load('../data/movielens/small_rating.csv')
log('Data loaded')


# convert data to Ratings objects
ratings = df.rdd.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
log('Data mapped to ratings')


# create train/test split
train, test = ratings.randomSplit([0.8, 0.2])
log('Train/Test split created')


# train model
K = 10
epochs = 10
model = ALS.train(train, K, epochs)
log('Training complete')


# get training MSE
x = train.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(x)\
    .map(lambda r: ((r[0], r[1]), r[2]))

ratingsAndPredictions = train\
    .map(lambda r: ((r[0], r[1]), r[2]))\
    .join(predictions)

mse = ratingsAndPredictions\
    .map(lambda r: (r[1][0] - r[1][1])**2)\
    .mean()

log(f'Train MSE: {mse}')


# get test MSE
x = test.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(x)\
    .map(lambda r: ((r[0], r[1]), r[2]))

ratingsAndPredictions = test\
    .map(lambda r: ((r[0], r[1]), r[2]))\
    .join(predictions)

mse = ratingsAndPredictions\
    .map(lambda r: (r[1][0] - r[1][1])**2)\
    .mean()

print(f'Test MSE: {mse}')

