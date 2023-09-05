import pandas as pd 
import numpy as np 
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import StringIndexer 
cols = ['useriD','StreamId', 'StreamerName', 'StartTime', 'EndTime']

"""
user - streamer - watch hours 
user - streamer - number of times used watched perticular streamer 
which is overall 2D
"""

def split_data(df, hold_out = 0.1):
    test = df.sample(frac = hold_out, replace = False)
    train = df[~df.index.isin(test.index)]
    return train, test

def build_watch_sparse(df):
    sparse_matrix = tf.SparseTensor(
        indices = df[['useriD','StreamerName']].values,
        values = df['ts'].values,
        dense_shape = [np.max(df['useriD']), np.max(df['StreamerName'])]
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
    df['combined'] = df['useriD'].astype(str) + '_' + df['StreamerName'].astype(str)

    df = df.sample(frac = 1, random_state = 42)

    train, test = split_data(df)

    train_combined = set(train['combined'])
    mask = test['combined'].isin(train_combined)
    filtered_test = test[mask]

    train_sparse = build_watch_sparse(train)
    test_sparse = build_watch_sparse(test)

    # sparse_matrix = build_watch_sparse(df)

    # print(sparse_matrix)
    return train_sparse, test_sparse


