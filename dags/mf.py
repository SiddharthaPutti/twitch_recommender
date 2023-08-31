# matrix factorization 

import tensorflow as tf
from preprocessData import *
# MSE
def sparse_loss(sparse_matrix, user_emb, streamer_emb):
    pred = tf.gather_nd(
        tf.matmul(user_emb, streamer_emb, transpose_b=True), sparse_matrix.indices
    )
    loss = tf.losses.mean_squared_error(sparse_matrix.values, predictions)
    return loss 

def split_data(data):
    pass

class colab_filtering():
    def __init__(self):
        pass

def build_model(data, emb_dim = 3, init_stddev = 1):
    train, test = split_data(data)
    train_sparse = build_watch_sparse(train)
    test_sparse = build_watch_sparse(test)

    U = tf.Variable(tf.random_normal(
        [train_sparse.dense_shape[0],emb_dim], stddev = init_stddev
    ))
    V = tf.Variable(tf.random_normal(
        [train_sparse.dense_shape[1], emb_dim], stddev = init_stddev
    ))

    train_loss = sparse_loss(train_sparse, U, V)
    test_loss = sparse_loss(test_sparse, U, V)

    



