import pandas as pd 
import numpy as np 
import tensorflow as tf
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import StringIndexer 
cols = ['useriD','StreamId', 'StreamerName', 'StartTime', 'EndTime']

"""
user - streamer - watch hours 
user - streamer - number of times used watched perticular streamer 
which is overall 2D
"""

def build_watch_sparse(df):
    sparse_matrix = tf.SparseTensor(
        indices = df[['useriD','StreamerName']].values,
        values = df['ts'].values,
        dense_shape = [np.unique(df['useriD']).shape[0], np.unique(df['StreamerName']).shape[0]]
    )
    return sparse_matrix

def preproc(df):

    # spark = SparkSession.builder.appName('preprocess_dag').getOrCreate()

    # df = spark.createDataFrame(df, cols)
    df = pd.DataFrame(df, columns = cols)
    df.useriD = pd.factorize(df.useriD)[0] +1
    df.StreamerName = pd.factorize(df.StreamerName)[0]+1

    df['ts'] = df['EndTime'] - df['StartTime']
    df = df.groupby(['useriD', 'StreamerName']).sum('ts').reset_index()

    # sparse_matrix = build_watch_sparse(df)

    # print(sparse_matrix)
    return df


