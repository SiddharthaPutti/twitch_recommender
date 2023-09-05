# matrix factorization 
import pandas as pd 
import numpy as np 
import tensorflow as tf
from preprocessData import *
tf.config.run_functions_eagerly(True)

# def split_data(data):
#     pass

# MSE
def sparse_loss(sparse_matrix, user_emb, streamer_emb):
    pred = tf.reduce_sum(
        tf.gather(user_emb, sparse_matrix.indices[:,0]) *
        tf.gather(streamer_emb, sparse_matrix.indices[:, 1]),
        axis = 1
    )

    loss = tf.losses.mean_squared_error(sparse_matrix.values, pred)
    return loss 


def embed_init(train, emb_dim = 3, init_stddev = 1):

    num_users, num_streamers = train.shape


    xavier_init = tf.initializers.GlorotNormal()

    U = tf.Variable(xavier_init(
        shape = (num_users+1,emb_dim)
    ))
    V = tf.Variable(xavier_init(
        shape = (num_streamers+1, emb_dim)
    ))

    return U,V



class colab_filtering():
    def __init__(self, train_sparse,test_sparse  U, V):
        self.train_sparse = train_sparse
        self.test_sparse = test_sparse
        self.U = U
        self.V =V

    def train_step(self,num_iter = 1000, lr = 3e-4):

        optimizer = tf.keras.optimizers.Adam(lr)

        for i in range(num_iter):
            with tf.GradientTape() as tape: 
                loss = sparse_loss(self.train_sparse, self.U,self.V)

            gradients = tape.gradient(loss, [self.U,self.V])
            optimizer.apply_gradients(zip(gradients, [self.U,self.V]))







    



